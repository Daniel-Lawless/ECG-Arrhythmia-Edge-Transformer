import argparse
import json
import logging
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from ecg_arrhythmia.data.build_xqrs_centered_dataset import load_split_record_names
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import (
    DEFAULT_WARMUP_CALLS,
    NANOSECONDS_PER_MILLISECOND,
    collect_sequences,
    environment_metadata,
    latency_summary_ms,
    throughput_sequences_per_second,
    time_inference,
    total_seconds,
)
from ecg_arrhythmia.evaluation.benchmark_quantized_inference import (
    DEFAULT_BENCHMARK_REPEATS,
    across_repeat_summary,
    benchmark_order,
    comparison_metrics,
    latency_comparison,
    repeat_summaries,
    size_comparison,
)
from ecg_arrhythmia.evaluation.validate_edge_streaming_runtime import health_snapshot
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE
from ecg_arrhythmia.telemetry.edge_sensors import (
    read_cpu_frequency_khz,
    read_cpu_governor,
)

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_FP32_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_INT8_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer_int8.onnx")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/deployment_evaluation/edge_onnx_benchmarking"
)
DEFAULT_FIGURES_DIR = Path("artifacts/figures/edge_onnx_benchmarking")

# The x86 Section 4.5 artifact, used for the cross-platform comparison.
DEFAULT_X86_BENCHMARK_PATH = Path(
    "artifacts/results/deployment_evaluation/onnx_benchmarking/"
    "fp32_vs_int8_benchmark.json"
)

ALL_RECORDS_RESULT_FILENAME = "raspberry_pi_fp32_vs_int8_benchmark.json"
ALL_RECORDS_RAW_FILENAME = "raspberry_pi_fp32_vs_int8_benchmark_raw.npz"

CLEAN_THROTTLED_STATE = "0x0"


# ---------------------------------------------------------------------
#                Memory-Conscious Per-Record Benchmarking
# ---------------------------------------------------------------------


def benchmark_records_individually(
    record_names: list[str],
    classifiers: dict[str, ONNXSequenceClassifier],
    chunk_size: int,
    warmup_calls: int,
    repeats: int,
    collect=collect_sequences,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    Benchmark both models over one record at a time.

    Each record's sequences are collected once, timed for every repeat in
    counterbalanced order by both classifiers over the identical
    BeatSequence objects, then released before the next record is
    collected. This helps memory stay smaller. Classifier sessions
    are constructed by the caller and reused throughout.
    """

    order = benchmark_order(repeats)
    record_benchmarks: list[dict[str, Any]] = []

    for record_name in record_names:
        # Collect the sequences for this record.
        sequences = collect(record_name, chunk_size)

        if not sequences:
            raise ValueError(f"Record {record_name} produced no sequences.")

        durations: dict[str, list[list[int]]] = {name: [] for name in classifiers}

        # For each pair
        for repeat_index, pair in enumerate(order):
            # for each model name in this pair
            for model_name in pair:
                logger.info(
                    "Record %s, repeat %d/%d: timing %s over %d sequences",
                    record_name,
                    repeat_index + 1,
                    repeats,
                    model_name.upper(),
                    len(sequences),
                )
                # Append the list of prediction times for each sequence
                # for this model
                # {
                #  fp32: [
                #          [duration ns for record ...], # this is pair 1
                #          [duration ns for record...], ... # then pair 2
                #        ]
                #  int8: [same then for int8]
                # }
                durations[model_name].append(
                    time_inference(
                        classifier=classifiers[model_name],
                        sequences=sequences,
                        warmup_calls=warmup_calls,
                    )
                )

        # Once we have went through each model pair, append the record name,
        # the number of sequences, and the list of list of durations that each
        # model pair produced.
        record_benchmarks.append(
            {
                "record_name": record_name,
                "num_sequences": len(sequences),
                "durations": durations,
            }
        )

        # Drop the current record before collecting the next one,
        # avoiding a brief overlap where both records occupy memory.
        del sequences

    return record_benchmarks, order


# ---------------------------------------------------------------------
#                     Pooling Across Records
# ---------------------------------------------------------------------


def global_repeat_durations(
    record_benchmarks: list[dict],
    model_name: str,
    repeat_index: int,
) -> list[int]:
    """One repeat's timings concatenated across every record."""

    return [
        value
        for benchmark in record_benchmarks
        for value in benchmark["durations"][model_name][repeat_index]
    ]


def pooled_durations(record_benchmarks: list[dict], model_name: str) -> list[int]:
    """Every timed inference for one model, across records and repeats."""

    return [
        value
        for benchmark in record_benchmarks
        for repeat in benchmark["durations"][model_name]
        for value in repeat
    ]


def model_statistics(
    record_benchmarks: list[dict],
    model_name: str,
    repeats: int,
    initialisation_ms: float,
) -> dict:
    """
    Per-repeat, across-repeat and pooled statistics for one model.

    Repeat i's global array is the concatenation of every record's repeat
    i, which is well defined because all records ran the same
    counterbalanced order. Pooled statistics use the raw durations, so
    records with more sequences carry proportionally more weight.
    """

    # Total number of sequences from all the records we benchmarked.
    num_sequences = sum(benchmark["num_sequences"] for benchmark in record_benchmarks)

    # One array per repeat, containing every sequence's inference duration
    # across all benchmarked records for this model.
    per_repeat_arrays = [
        global_repeat_durations(record_benchmarks, model_name, repeat_index)
        for repeat_index in range(repeats)
    ]

    # calculates the min, max, mean, median, p95, and throughput
    # across each repeat.
    per_repeat = repeat_summaries(per_repeat_arrays, num_sequences)

    return {
        "classifier_initialisation_ms": initialisation_ms,
        "per_repeat": per_repeat,
        "across_repeats": across_repeat_summary(per_repeat),
        # pooled_durations(...) returns every individual inference
        # duration for that model across every benchmarked record,
        # every repeat, every sequence within each repeat. Then we
        # calculate min, max, mean, median, p95, and throughput
        # for that.
        "pooled_all_repeats": latency_summary_ms(
            pooled_durations(record_benchmarks, model_name)
        ),
    }


def per_record_summary(record_benchmark: dict) -> dict:
    """Pooled per-record latency for both models, with comparison."""

    # Create this record's result dictionary and initialise it with its
    # name and number of sequences.
    result: dict = {
        "record_name": record_benchmark["record_name"],
        "num_sequences": record_benchmark["num_sequences"],
    }

    # For each model and its repeat durations.
    for model_name in record_benchmark["durations"]:
        # Pool every sequence's inference duration across all repeats
        # for this model and this record.
        pooled = [
            duration
            for repeat in record_benchmark["durations"][model_name]
            for duration in repeat
        ]

        # Calculate the min, max, mean, median and p95 latency
        # across all repeats for this record.
        stats = latency_summary_ms(pooled)

        # Calculate the throughput across all repeats for this record.
        stats["throughput"] = throughput_sequences_per_second(
            len(pooled),
            total_seconds(pooled),
        )

        # Add this model's statistics to the record result.
        result[model_name] = stats

    # Get the FP32 and INT8 summaries for easier comparison.
    fp32 = result["fp32"]
    int8 = result["int8"]

    # Add a comparison between FP32 and INT8 for this record.
    # This includes the mean latency comparison, p95 latency speedup,
    # and percentage change in throughput.
    result["comparison"] = {
        "mean_latency": latency_comparison(
            fp32["mean"],
            int8["mean"],
        ),
        "p95_latency_speedup": (fp32["p95"] / int8["p95"]),
        "throughput_change_percentage": (
            (int8["throughput"] - fp32["throughput"]) / fp32["throughput"] * 100.0
        ),
    }

    return result


# ---------------------------------------------------------------------
#                   Cross-Platform Comparison (x86)
# ---------------------------------------------------------------------


def load_x86_benchmark(path: Path) -> dict | None:
    """Load the Section 4.5 x86 result, degrading to None if absent."""

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.warning("x86 Section 4.5 benchmark unavailable at %s", path)
        return None


def cross_platform_comparison(
    x86_result: dict | None,
    pi_fp32_mean_ms: float,
    pi_int8_mean_ms: float,
    pi_int8_speedup: float,
) -> dict | None:
    """
    Compare the Pi run with the measured x86 Section 4.5 run.

    Purely descriptive: ratios between two measured environments, with
    no claim about why either platform behaves as it does.
    """

    if x86_result is None:
        return None

    try:
        # Extract the mean of the mean across the repeats, the same
        # values we passed into this function from the pi, but for the
        # local machine. Same for the mean latency speed up.
        x86_fp32_mean = x86_result["fp32"]["across_repeats"]["mean"]["mean"]
        x86_int8_mean = x86_result["int8"]["across_repeats"]["mean"]["mean"]
        x86_speedup = x86_result["comparison"]["mean_latency"]["speedup"]
    except (KeyError, TypeError):
        logger.warning("x86 benchmark JSON did not have the expected shape")
        return None

    return {
        "x86_fp32_mean_latency_ms": x86_fp32_mean,
        "pi_fp32_mean_latency_ms": pi_fp32_mean_ms,
        "fp32_pi_over_x86_latency_ratio": pi_fp32_mean_ms / x86_fp32_mean,
        "x86_int8_mean_latency_ms": x86_int8_mean,
        "pi_int8_mean_latency_ms": pi_int8_mean_ms,
        "int8_pi_over_x86_latency_ratio": pi_int8_mean_ms / x86_int8_mean,
        "x86_int8_vs_fp32_speedup": x86_speedup,
        "pi_int8_vs_fp32_speedup": pi_int8_speedup,
    }


# ---------------------------------------------------------------------
#                    Pi Health Context (untimed)
# ---------------------------------------------------------------------


def benchmark_health_snapshot() -> dict:
    """Section 5.1's health snapshot plus CPU frequency context."""

    # This returns the total and available ram of the pi as well
    # as the temperature and throttle flag.
    snapshot = health_snapshot()
    # Returns what governor the CPU is using
    snapshot["cpu_governor"] = read_cpu_governor()
    # Returns the frequency of the CPU clock
    snapshot["cpu_frequency_khz"] = read_cpu_frequency_khz()

    return snapshot


def timing_interpretation_warning(
    throttled_before: str | None,
    throttled_after: str | None,
) -> str | None:
    """
    Flag timings that may be thermally contaminated.

    Only the hardware's own throttling state is consulted; no invented
    temperature threshold. Unknown states produce no warning because
    absence of evidence is not evidence of throttling.
    """

    abnormal = {
        label: value
        for label, value in (
            ("before", throttled_before),
            ("after", throttled_after),
        )
        if value is not None and value != CLEAN_THROTTLED_STATE
    }

    if not abnormal:
        return None

    flags = ", ".join(f"{label}={value}" for label, value in abnormal.items())

    return (
        f"Throttling flags were not clean ({flags}); latency figures may "
        "be thermally contaminated and should be interpreted with caution."
    )


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def _timed_classifier(model_path: Path) -> tuple[ONNXSequenceClassifier, float]:
    start_ns = perf_counter_ns()
    classifier = ONNXSequenceClassifier(model_path)

    return classifier, (perf_counter_ns() - start_ns) / NANOSECONDS_PER_MILLISECOND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FP32 and INT8 model-stage inference on the Raspberry "
            "Pi, one record at a time, using the Section 4 timing boundary."
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
    parser.add_argument("--warmup-calls", type=int, default=DEFAULT_WARMUP_CALLS)
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=DEFAULT_BENCHMARK_REPEATS,
    )
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--x86-benchmark-path",
        type=Path,
        default=DEFAULT_X86_BENCHMARK_PATH,
    )
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--write-plots",
        action="store_true",
        help="Also render figures (development machine only; needs matplotlib).",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Extract CL arguments
    args = parse_args()

    # Fails before any model work if the repeat count is unusable.
    benchmark_order(args.benchmark_repeats)

    # if we set the all-validation-records flag it loads of the patient ids from the
    # validation set
    if args.all_validation_records:
        record_names = load_split_record_names(args.split_summary_path, "val")
        result_filename = ALL_RECORDS_RESULT_FILENAME
        raw_filename = ALL_RECORDS_RAW_FILENAME
    # Else we just use the record name provided explicitly in the CL
    else:
        record_names = [args.record_name]
        result_filename = f"record_{args.record_name}_fp32_vs_int8_benchmark.json"
        raw_filename = f"record_{args.record_name}_fp32_vs_int8_benchmark_raw.npz"

    # Load the ONNX classifers and time their initialisation
    fp32_classifier, fp32_initialisation_ms = _timed_classifier(args.fp32_model_path)
    int8_classifier, int8_initialisation_ms = _timed_classifier(args.int8_model_path)
    classifiers = {"fp32": fp32_classifier, "int8": int8_classifier}

    # Returns the available and total ram, the temp, throttle flag,
    # CPU governor, and CPU clock before benchmarking.
    health_before = benchmark_health_snapshot()

    # returns the models prediction latency for each record, along with the
    # ordering that was used
    record_benchmarks, order = benchmark_records_individually(
        record_names=record_names,
        classifiers=classifiers,
        chunk_size=args.chunk_size,
        warmup_calls=args.warmup_calls,
        repeats=args.benchmark_repeats,
    )

    # Returns the available and total ram, the temp, throttle flag,
    # CPU governor, and CPU clock after benchmarking.
    health_after = benchmark_health_snapshot()

    # Returns the model intialisation, min, max, mean, median, p95,
    # and throughput across each repeat, and pooled results,
    # and the mean, median etc of the min, max, mean, median,...
    # for each repeat
    model_results = {
        "fp32": model_statistics(
            record_benchmarks,
            "fp32",
            args.benchmark_repeats,
            fp32_initialisation_ms,
        ),
        "int8": model_statistics(
            record_benchmarks,
            "int8",
            args.benchmark_repeats,
            int8_initialisation_ms,
        ),
    }
    # Returns the delta, percentage change, and speedup across
    # the mean_latency, median_latency, p95_latency, and throughput
    # for the repeats.
    comparison = comparison_metrics(
        model_results["fp32"]["across_repeats"],
        model_results["int8"]["across_repeats"],
    )
    # checks whether the Raspberry Pi reported any throttling
    # before or after the benchmark. if so, it returns a warning string
    # saying the timing results may have been affected by heat/power
    # conditions.
    warning = timing_interpretation_warning(
        health_before["throttled"],
        health_after["throttled"],
    )
    # Compares the mean across the means of each record and
    # the mean latency speed on the PI to the local machine
    # benchmarks we obtained.
    cross = cross_platform_comparison(
        load_x86_benchmark(args.x86_benchmark_path),
        model_results["fp32"]["across_repeats"]["mean"]["mean"],
        model_results["int8"]["across_repeats"]["mean"]["mean"],
        comparison["mean_latency"]["speedup"],
    )

    result = {
        "model_paths": {
            "fp32": str(args.fp32_model_path),
            "int8": str(args.int8_model_path),
        },
        "size": size_comparison(
            args.fp32_model_path.stat().st_size,
            args.int8_model_path.stat().st_size,
        ),
        "benchmark": {
            "target": "raspberry_pi",
            "record_names": record_names,
            "num_records": len(record_names),
            "num_sequences": sum(
                benchmark["num_sequences"] for benchmark in record_benchmarks
            ),
            "sequences_per_record": {
                benchmark["record_name"]: benchmark["num_sequences"]
                for benchmark in record_benchmarks
            },
            "chunk_size": args.chunk_size,
            "warmup_calls": args.warmup_calls,
            "benchmark_repeats": args.benchmark_repeats,
            "benchmark_order": [list(pair) for pair in order],
            "per_record_execution": True,
            "provider": fp32_classifier.providers[0],
            "session_configuration": "onnxruntime defaults, identical for both",
        },
        "fp32": model_results["fp32"],
        "int8": model_results["int8"],
        "per_record": [
            per_record_summary(benchmark) for benchmark in record_benchmarks
        ],
        "comparison": comparison,
        "cross_platform": cross,
        "hardware_health": {"before": health_before, "after": health_after},
        "timing_interpretation_warning": warning,
        "environment": environment_metadata(fp32_classifier.providers),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / result_filename).open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    raw_arrays = {
        f"{model_name}_{benchmark['record_name']}_repeat_{index}_ns": np.asarray(
            repeat,
            dtype=np.int64,
        )
        for benchmark in record_benchmarks
        for model_name, repeats in benchmark["durations"].items()
        for index, repeat in enumerate(repeats)
    }
    np.savez_compressed(args.output_dir / raw_filename, **raw_arrays)

    logger.info("Wrote benchmark to %s", args.output_dir / result_filename)

    if args.write_plots:
        # Lazy import keeps matplotlib off the Pi execution path.
        from ecg_arrhythmia.evaluation.onnx_benchmark_plots import (
            write_benchmark_figures,
        )

        written = write_benchmark_figures(
            fp32_summary=model_results["fp32"]["across_repeats"],
            int8_summary=model_results["int8"]["across_repeats"],
            size=result["size"],
            per_record=result["per_record"],
            figures_dir=args.figures_dir,
        )
        for path in written:
            logger.info("Wrote figure %s", path)


if __name__ == "__main__":
    main()
