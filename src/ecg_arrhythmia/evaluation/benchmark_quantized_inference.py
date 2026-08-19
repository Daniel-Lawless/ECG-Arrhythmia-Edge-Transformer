import argparse
import json
import logging
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from ecg_arrhythmia.data.build_xqrs_centered_dataset import load_split_record_names
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import (
    DEFAULT_WARMUP_CALLS,
    NANOSECONDS_PER_MILLISECOND,
    collect_sequences,
    environment_metadata,
    file_size_mib,
    latency_summary_ms,
    throughput_sequences_per_second,
    time_inference,
    total_seconds,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/results/deployment_evaluation/onnx_benchmarking")
DEFAULT_FIGURES_DIR = Path("artifacts/figures/onnx_benchmarking")
DEFAULT_FP32_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_INT8_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer_int8.onnx")

RESULT_FILENAME = "fp32_vs_int8_benchmark.json"
RAW_FILENAME = "fp32_vs_int8_benchmark_raw.npz"

DEFAULT_BENCHMARK_REPEATS = 5


# ---------------------------------------------------------------------
#                        Benchmark Orchestration
# ---------------------------------------------------------------------


def benchmark_order(repeats: int) -> list[tuple[str, str]]:
    """
    Deterministic counterbalanced model order for each repeat.

    Alternating which model runs first cancels simple thermal and load
    drift without introducing nondeterministic shuffling.
    """

    if repeats < 1:
        raise ValueError("At least one benchmark repeat is required.")

    return [
        ("fp32", "int8") if repeat % 2 == 0 else ("int8", "fp32")
        for repeat in range(repeats)
    ]


def verify_identical_sequences(
    first: list[BeatSequence],
    second: list[BeatSequence],
) -> None:
    """
    Fail loudly if two sequence collections are not the same inputs.

    The benchmark passes one list to both models, but any caller holding
    two references must prove count, order and target peaks all match.
    """

    if len(first) != len(second):
        raise ValueError(
            f"Sequence collections differ in length: {len(first)} versus {len(second)}."
        )

    for position, (a, b) in enumerate(zip(first, second, strict=True)):
        if a.target_peak_index != b.target_peak_index:
            raise ValueError(
                f"Sequence collections diverge at position {position}: "
                f"target peaks {a.target_peak_index} versus "
                f"{b.target_peak_index}."
            )


def run_repeated_benchmark(
    classifiers: dict[str, ONNXSequenceClassifier],
    sequences: list[BeatSequence],
    warmup_calls: int,
    repeats: int,
) -> tuple[dict[str, list[list[int]]], list[tuple[str, str]]]:
    """
    Time both models over the same sequences for every repeat.

    Each model is warmed up before each timed pass with the same leading
    sequences, and sessions are never rebuilt between repeats, so this
    measures steady-state inference under identical conditions.
    """

    # gives a list of tuples that switches the model order. We do this
    # to reduce the bias of running one model first over another.
    order = benchmark_order(repeats)
    durations: dict[str, list[list[int]]] = {name: [] for name in classifiers}

    # Extract a model pair and its index
    for repeat_index, pair in enumerate(order):
        # iterate through the model pair
        for model_name in pair:
            logger.info(
                "Repeat %d/%d: timing %s over %d sequences",
                repeat_index + 1,
                repeats,
                model_name.upper(),
                len(sequences),
            )
            # Each run appends a list of how long each prediction took for
            # each sequence in sequences
            # "fp32": [[durantion ns run 1], ... , [duration ns run order]]
            # "int8": [[durantion ns run 1], ... , [duration ns run order]]
            durations[model_name].append(
                time_inference(
                    classifier=classifiers[model_name],
                    sequences=sequences,
                    warmup_calls=warmup_calls,
                )
            )

    return durations, order


# ---------------------------------------------------------------------
#                          Summary Statistics
# ---------------------------------------------------------------------


def repeat_summaries(
    repeat_durations: list[list[int]],
    num_sequences: int,
) -> list[dict]:
    """Latency and throughput summary for each individual repeat."""

    summaries = []

    # For each run
    for durations in repeat_durations:
        # Calculates the minimum, maximum, mean, median, and p95
        # of this duration
        summary = latency_summary_ms(durations)

        # Calculates the throughput of this duration. This is the number
        # of sequences predicted per second
        summary["throughput"] = throughput_sequences_per_second(
            num_sequences,
            total_seconds(durations),
        )
        # Each summary dictionary summaries one duration. Summaries contains
        # each duration summary.
        summaries.append(summary)

    return summaries


def summarise_across_repeats(values: list[float]) -> dict[str, float]:
    """Spread of one headline metric across repeats."""

    if not values:
        raise ValueError("At least one repeat value is required.")

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def across_repeat_summary(per_repeat: list[dict]) -> dict:
    """
    Summarise each headline metric across repeats.

    Per-repeat p95 values are summarised as p95 values; they are not
    averaged into a pretend pooled p95. Pooled statistics over every
    timed inference are reported separately by the caller.
    """

    # This will calculate the mean, median, minimum, and maximum of each
    # metric across each repeat.
    return {
        metric: summarise_across_repeats([repeat[metric] for repeat in per_repeat])
        for metric in ("mean", "median", "p95", "maximum", "minimum", "throughput")
    }


# ---------------------------------------------------------------------
#                         Comparison Metrics
# ---------------------------------------------------------------------


def latency_comparison(fp32_ms: float, int8_ms: float) -> dict[str, float]:
    """
    Signed delta, percentage change and speedup for one latency metric.

    Speedup above 1.0 means INT8 is faster, negative change percentage
    means INT8 takes less time.
    """

    if fp32_ms <= 0 or int8_ms <= 0:
        raise ValueError("Latency values must be positive.")

    return {
        "delta_ms": int8_ms - fp32_ms,
        "change_percentage": (int8_ms - fp32_ms) / fp32_ms * 100.0,
        "speedup": fp32_ms / int8_ms,
    }


def throughput_comparison(fp32_value: float, int8_value: float) -> dict[str, float]:
    """Signed delta, percentage change and speedup for throughput."""

    if fp32_value <= 0 or int8_value <= 0:
        raise ValueError("Throughput values must be positive.")

    return {
        "delta": int8_value - fp32_value,
        "change_percentage": (int8_value - fp32_value) / fp32_value * 100.0,
        "speedup": int8_value / fp32_value,
    }


def size_comparison(fp32_bytes: int, int8_bytes: int) -> dict:
    """Model file sizes with reduction and compression ratio."""

    if fp32_bytes <= 0 or int8_bytes <= 0:
        raise ValueError("Model sizes must be positive.")

    return {
        "fp32_bytes": fp32_bytes,
        "fp32_mib": file_size_mib(fp32_bytes),
        "int8_bytes": int8_bytes,
        "int8_mib": file_size_mib(int8_bytes),
        "reduction_mib": file_size_mib(fp32_bytes - int8_bytes),
        "reduction_percentage": (fp32_bytes - int8_bytes) / fp32_bytes * 100.0,
        "compression_ratio": fp32_bytes / int8_bytes,
    }


def comparison_metrics(fp32_summary: dict, int8_summary: dict) -> dict:
    """
    Compare the across-repeat mean of each headline metric.

    Uses the mean across repeats rather than a single pass, so the
    comparison reflects steady-state behaviour.
    """

    # returns the change, precentage change, and speedup for
    # each metric. 
    return {
        "mean_latency": latency_comparison(
            fp32_summary["mean"]["mean"],
            int8_summary["mean"]["mean"],
        ),
        "median_latency": latency_comparison(
            fp32_summary["median"]["mean"],
            int8_summary["median"]["mean"],
        ),
        "p95_latency": latency_comparison(
            fp32_summary["p95"]["mean"],
            int8_summary["p95"]["mean"],
        ),
        "throughput": throughput_comparison(
            fp32_summary["throughput"]["mean"],
            int8_summary["throughput"]["mean"],
        ),
    }


# ---------------------------------------------------------------------
#                          Per-Record Results
# ---------------------------------------------------------------------


def per_record_results(
    boundaries: list[tuple[str, int, int]],
    durations: dict[str, list[list[int]]],
) -> list[dict]:
    """
    Per-record latency from the pooled runs.

    The pooled sequence list preserves record order, so each record's
    timings are a slice of every repeat's duration array; statistics
    pool that record's slices across all repeats.
    """

    results = []

    # For each record and its boundaries start and stop index
    for record_name, start, stop in boundaries:
        # Create this records dictionary and intialise it with its name
        # and number of sequences
        record_result: dict = {
            "record_name": record_name,
            "num_sequences": stop - start,
        }

        # For each model and the repeats durations
        for model_name, repeats in durations.items():
            # Extract the values from each repeats slice that correspond 
            # to this record
            pooled = [
                duration
                for repeat in repeats
                for duration in repeat[start:stop]
            ]
            # Calculate the min, max, mean, median, p95 of this slice
            summary = latency_summary_ms(pooled)
            # Calculate the throughput of this slice.
            summary["throughput"] = throughput_sequences_per_second(
                len(pooled),
                total_seconds(pooled),
            )
            # Append this models result for this record 
            record_result[model_name] = summary

        # Also adds a comparison key to the record result dict that will hold
        # the speedup between the mean duration of the fp32 predictions for this 
        # record and the mean duration of the int8 predictions for this record.
        # Similarly for p95. It will also hold the throughput percentage change
        # between fp32 and int8 on this record. 
        record_result["comparison"] = {
            "mean_latency_speedup": (
                record_result["fp32"]["mean"] / record_result["int8"]["mean"]
            ),
            "p95_latency_speedup": (
                record_result["fp32"]["p95"] / record_result["int8"]["p95"]
            ),
            "throughput_change_percentage": (
                (
                    record_result["int8"]["throughput"]
                    - record_result["fp32"]["throughput"]
                )
                / record_result["fp32"]["throughput"]
                * 100.0
            ),
        }
        # Appends this records record_result to the results dict,
        # which will be a list of dicts, each holding summaries of 
        # the metrics per record
        results.append(record_result)

    return results


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def _timed_classifier(model_path: Path) -> tuple[ONNXSequenceClassifier, float]:
    """Construct one classifier, timing initialisation separately."""

    start_ns = perf_counter_ns()
    classifier = ONNXSequenceClassifier(model_path)
    # Convert to milliseconds
    initialisation_ms = (perf_counter_ns() - start_ns) / NANOSECONDS_PER_MILLISECOND

    return classifier, initialisation_ms


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FP32 and INT8 under identical production inference "
            "conditions and save comparative performance evidence."
        )
    )
    parser.add_argument(
        "--fp32-model-path",
        type=Path,
        default=DEFAULT_FP32_MODEL_PATH,
    )
    parser.add_argument(
        "--int8-model-path",
        type=Path,
        default=DEFAULT_INT8_MODEL_PATH,
    )
    parser.add_argument("--record-name", type=str, default=DEFAULT_RECORD_NAME)
    parser.add_argument(
        "--all-validation-records",
        action="store_true",
        help="Benchmark every record in the validation split.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--warmup-calls",
        type=int,
        default=DEFAULT_WARMUP_CALLS,
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=DEFAULT_BENCHMARK_REPEATS,
        help="Full timed passes per model; must be at least 1.",
    )
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--write-plots",
        action="store_true",
        help="Save latency, throughput, size and per-record figures.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Extract CL arguments.
    args = parse_args()

    # Fails before any model work if the repeat count is unusable.
    order = benchmark_order(args.benchmark_repeats)

    if args.all_validation_records:
        record_names = load_split_record_names(args.split_summary_path, "val")
    else:
        record_names = [args.record_name]

    # One collection pass; both models consume this identical list.
    sequences: list[BeatSequence] = []
    boundaries: list[tuple[str, int, int]] = []

    # For each record
    for record_name in record_names:
        # Produce the sequences from this record from the streaming pipeline
        record_sequences = collect_sequences(record_name, args.chunk_size)

        # Gives the boundary of each record before the next record sequences
        # begin
        boundaries.append(
            (record_name, len(sequences), len(sequences) + len(record_sequences))
        )
        # Add this records sequences to the overall sequences
        sequences.extend(record_sequences)

    if not sequences:
        raise SystemExit("The selected records produced no sequences.")

    # Create the classifier and time its intialisation
    fp32_classifier, fp32_initialisation_ms = _timed_classifier(args.fp32_model_path)
    int8_classifier, int8_initialisation_ms = _timed_classifier(args.int8_model_path)

    # Returns the duration of each sequence prediction in sequences for each model
    # over repeats amount of runs. Also returns the order in which the models
    # where run.
    durations, order = run_repeated_benchmark(
        classifiers={"fp32": fp32_classifier, "int8": int8_classifier},
        sequences=sequences,
        warmup_calls=args.warmup_calls,
        repeats=args.benchmark_repeats,
    )

    model_results = {}

    for model_name, initialisation_ms in (
        ("fp32", fp32_initialisation_ms),
        ("int8", int8_initialisation_ms),
    ):
        # returns a dictionary describing the min, max, median, mean, p95, 
        # and throughout of each duration
        per_repeat = repeat_summaries(durations[model_name], len(sequences))

        # Combines all FP32 per prediction durations into a list. Same
        # for INT8.
        pooled = [
            duration
            for repeat in durations[model_name]
            for duration in repeat
        ]
        # Computes min, max, median, mean, and p95 metrics across all repeats.
        pooled_summary = latency_summary_ms(pooled)

        # Combines results into one dictionary. It includes the classifiers 
        # intialisation_ms, its per repeat metrics, its summary across repeat,
        # and metrics across all repeats
        model_results[model_name] = {
            "classifier_initialisation_ms": initialisation_ms,
            "per_repeat": per_repeat,
            "across_repeats": across_repeat_summary(per_repeat),
            "pooled_all_repeats": pooled_summary,
        }

    # Returns the delta, percentage change, and speedup across each
    # across repeats metric. 
    comparison = comparison_metrics(
        model_results["fp32"]["across_repeats"],
        model_results["int8"]["across_repeats"],
    )
    # Returns size comparison metrics, such as compression ratio,
    # file size change, size in Bytes and MIB etc
    size = size_comparison(
        args.fp32_model_path.stat().st_size,
        args.int8_model_path.stat().st_size,
    )
    # Change in initialisation time
    initialisation_delta = int8_initialisation_ms - fp32_initialisation_ms

    result = {
        "model_paths": {
            "fp32": str(args.fp32_model_path),
            "int8": str(args.int8_model_path),
        },
        "size": size,
        "benchmark": {
            "record_names": record_names,
            "num_records": len(record_names),
            "num_sequences": len(sequences),
            "chunk_size": args.chunk_size,
            "warmup_calls": args.warmup_calls,
            "benchmark_repeats": args.benchmark_repeats,
            "benchmark_order": [list(pair) for pair in order],
            "provider": fp32_classifier.providers[0],
            "session_configuration": "onnxruntime defaults, identical for both",
        },
        "fp32": model_results["fp32"],
        "int8": model_results["int8"],
        "initialisation_comparison": {
            "delta_ms": initialisation_delta,
            "change_percentage": (
                initialisation_delta / fp32_initialisation_ms * 100.0
            ),
        },
        "per_record": per_record_results(boundaries, durations),
        "comparison": comparison,
        "environment": environment_metadata(fp32_classifier.providers),
    }

    _write_json(result, args.output_dir / RESULT_FILENAME)

    if args.write_plots:
        from ecg_arrhythmia.evaluation.onnx_benchmark_plots import (
            write_benchmark_figures,
        )

        written = write_benchmark_figures(
            fp32_summary=model_results["fp32"]["across_repeats"],
            int8_summary=model_results["int8"]["across_repeats"],
            size=size,
            per_record=result["per_record"],
            figures_dir=args.figures_dir,
        )
        for path in written:
            logger.info("Wrote figure %s", path)

    logger.info("Wrote benchmark to %s", args.output_dir / RESULT_FILENAME)


if __name__ == "__main__":
    main()
