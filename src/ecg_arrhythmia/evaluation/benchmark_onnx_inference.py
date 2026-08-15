import argparse
import json
import logging
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import onnxruntime as ort

from ecg_arrhythmia.data.build_xqrs_centered_dataset import load_split_record_names
from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplaySource
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/results/deployment_evaluation/onnx_benchmarking")
DEFAULT_ONNX_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
RESULT_FILENAME = "fp32_onnx_benchmark.json"

# Enough calls for ONNX Runtime to finish its lazy first-call allocation.
# We use warmup calls since the first time we pass data through a ML model
# the framework often has to allocate buffers, load weights, etc,
# which can make the first predictions slower, which would skew
# our benchmark metrics by making it appear slower than its real
# # steady-state performance
DEFAULT_WARMUP_CALLS = 100

BYTES_PER_MIB = 1024 * 1024
NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000


# ---------------------------------------------------------------------
#                          Measurement Helpers
# ---------------------------------------------------------------------


def file_size_mib(size_bytes: int) -> float:
    """Convert a byte count to mebibytes."""

    if size_bytes < 0:
        raise ValueError("File size must not be negative.")

    return size_bytes / BYTES_PER_MIB


def effective_warmup_calls(warmup_calls: int, num_sequences: int) -> int:
    """Warm-up cannot use more sequences than were actually collected."""

    if warmup_calls < 0:
        raise ValueError("Warm-up call count must not be negative.")

    return min(warmup_calls, num_sequences)


def latency_summary_ms(durations_ns: Sequence[int]) -> dict[str, float]:
    """
    Summarise per-call inference latency in milliseconds.

    Timing is kept in integer nanoseconds until this point so no
    precision is lost accumulating small durations.
    """

    if len(durations_ns) == 0:
        raise ValueError("Cannot summarise latency without any timed inferences.")

    # Converts durations_ns to a np.array and convers each nanosecond measurements
    # to miliseconds through broadcasting
    durations_ms = (
        np.asarray(durations_ns, dtype=np.float64) / NANOSECONDS_PER_MILLISECOND
    )

    return {
        # The fastest time a prediction took
        "minimum": float(durations_ms.min()),
        # The average time each prediction took
        "mean": float(durations_ms.mean()),
        # 50% of prediction timings are below or equal this value
        "median": float(np.percentile(durations_ms, 50)),
        # 95% of predictions timings are below or equal this value
        "p95": float(np.percentile(durations_ms, 95)),
        # The longest time it took the a prediction
        "maximum": float(durations_ms.max()),
    }


def total_seconds(durations_ns: Sequence[int]) -> float:
    """
    Total timed inference duration, in seconds.

    The float() wrap keeps the return a plain Python float even when a
    numpy array is summed, so downstream JSON writing never sees a
    numpy scalar.
    """

    return float(sum(durations_ns)) / NANOSECONDS_PER_SECOND


def throughput_sequences_per_second(
    num_sequences: int,
    timed_seconds: float,
) -> float:
    """
    Sequences classified per second of timed inference.

    Deliberately excludes preprocessing and session creation, so this is
    the model stage's ceiling rather than the pipeline's rate.
    """

    if num_sequences < 0:
        raise ValueError("Sequence count must not be negative.")

    if timed_seconds <= 0:
        raise ValueError("Timed duration must be positive to compute throughput.")

    return num_sequences / timed_seconds


# ---------------------------------------------------------------------
#                         Collecting Real Inputs
# ---------------------------------------------------------------------


def collect_sequences(record_name: str, chunk_size: int) -> list[BeatSequence]:
    """
    Replay one record through the streaming pipeline and keep its output.

    The engine is driven directly rather than through StreamingPredictor,
    so no inference runs while the benchmark inputs are gathered.
    """

    # Load the record
    signals, fields, _ = load_record(record_name=record_name)
    # Select MLII
    signal, _ = select_signal_channel(signals=signals, fields=fields)

    # Populate ReplaySource with this record
    source = ReplaySource(
        signal=signal,
        sampling_rate=float(fields["fs"]),
        chunk_size=chunk_size,
        record_name=record_name,
    )
    # Create the streaming engine
    engine = StreamingEngine()
    # Setup the engine to receive the record.
    engine.start_record(record_name=record_name)

    sequences: list[BeatSequence] = []

    # For each chunk in the replay source
    for chunk in source.iter_chunks():
        # process the chunk through the engine and add it to sequences
        sequences.extend(engine.process_chunk(chunk))

    # Add the flush sequences to sequences
    sequences.extend(engine.flush())

    logger.info("Record %s produced %d sequences", record_name, len(sequences))

    return sequences


# ---------------------------------------------------------------------
#                               Timing
# ---------------------------------------------------------------------


def time_inference(
    classifier: ONNXSequenceClassifier,
    sequences: Sequence[BeatSequence],
    warmup_calls: int = DEFAULT_WARMUP_CALLS,
) -> list[int]:
    """
    Time one predict call per sequence and return the durations in ns.
    Performs warmup predictions to all for lazy intialization, buffer
    allocation, etc.
    """

    if not sequences:
        raise ValueError("At least one sequence is required to benchmark.")

    # Run the model on a few warmup calls first to get it steady
    for index in range(effective_warmup_calls(warmup_calls, len(sequences))):
        classifier.predict(sequences[index])

    durations_ns: list[int] = []

    # Now that the mdoel is warmed up, for each sequence
    for sequence in sequences:
        # Start the nano seconds counter
        start_ns = perf_counter_ns()
        # Make a prediction
        classifier.predict(sequence)
        # Append its time to durations_ns
        durations_ns.append(perf_counter_ns() - start_ns)

    return durations_ns


# ---------------------------------------------------------------------
#                               Benchmark
# ---------------------------------------------------------------------


def environment_metadata(providers: Sequence[str]) -> dict:
    """Lightweight machine and runtime details, from the standard library."""

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "onnxruntime_version": ort.__version__,
        "execution_providers": list(providers),
    }


def benchmark_records(
    record_names: list[str],
    onnx_model_path: Path = DEFAULT_ONNX_MODEL_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    warmup_calls: int = DEFAULT_WARMUP_CALLS,
) -> dict:
    """Collect real sequences, then benchmark FP32 inference over them."""

    if not record_names:
        raise ValueError("At least one record name must be supplied.")

    sequences: list[BeatSequence] = []
    sequences_per_record: dict[str, int] = {}

    # For each record
    for record_name in record_names:
        # Collect the sequences generated by the streaming engine for
        # this record
        record_sequences = collect_sequences(record_name, chunk_size)
        # Store the record name and its corresponding number of sequences
        sequences_per_record[record_name] = len(record_sequences)
        # Add this record sequences to the total sequences
        sequences.extend(record_sequences)

    if not sequences:
        raise ValueError("The selected records produced no sequences to benchmark.")

    # Timed separately because it happens once per process and the
    # session is then reused for every prediction.
    initialisation_start_ns = perf_counter_ns()
    classifier = ONNXSequenceClassifier(onnx_model_path)
    initialisation_ns = perf_counter_ns() - initialisation_start_ns

    logger.info(
        "Benchmarking %d sequences after %d warm-up calls",
        len(sequences),
        effective_warmup_calls(warmup_calls, len(sequences)),
    )

    # Return a list of how long each prediction took
    durations_ns = time_inference(
        classifier=classifier,
        sequences=sequences,
        warmup_calls=warmup_calls,
    )
    # Convert total run time from nanoseconds to seconds
    timed_seconds = total_seconds(durations_ns)
    # .stat() tells us statisitcs about the file, i.e., when it was
    # created, its size, etc. Then .st_size tells us the file size in bytes
    size_bytes = onnx_model_path.stat().st_size

    return {
        "model": {
            "path": str(onnx_model_path),
            "precision": "fp32",
            "size_bytes": size_bytes,
            "size_mib": file_size_mib(size_bytes),
        },
        "benchmark": {
            "record_names": list(record_names),
            "num_records": len(record_names),
            "num_sequences": len(sequences),
            "sequences_per_record": sequences_per_record,
            "chunk_size": chunk_size,
            "warmup_calls": effective_warmup_calls(warmup_calls, len(sequences)),
            "provider": classifier.providers[0],
            "classifier_initialisation_ms": (
                initialisation_ns / NANOSECONDS_PER_MILLISECOND
            ),
            "total_timed_inference_seconds": timed_seconds,
            "latency_ms": latency_summary_ms(durations_ns),
            "throughput_sequences_per_second": throughput_sequences_per_second(
                len(sequences),
                timed_seconds,
            ),
        },
        "environment": environment_metadata(classifier.providers),
    }


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FP32 ONNX inference for one production batch-of-one "
            "streaming sequence."
        )
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
        help="Untimed predict calls made before measurement begins.",
    )
    parser.add_argument(
        "--onnx-model-path",
        type=Path,
        default=DEFAULT_ONNX_MODEL_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Load our arguments
    args = parse_args()

    # Extract all validation records if cli argument is provided
    if args.all_validation_records:
        record_names = load_split_record_names(args.split_summary_path, "val")
    else:
        # Else load the record that was explicity passed.
        record_names = [args.record_name]

    # returns model, benchmark, and environment results
    result = benchmark_records(
        record_names=record_names,
        onnx_model_path=args.onnx_model_path,
        chunk_size=args.chunk_size,
        warmup_calls=args.warmup_calls,
    )

    # Write the results to output_dir / result_filename
    result_path = args.output_dir / RESULT_FILENAME
    _write_json(result, result_path)

    logger.info("Wrote benchmark results to %s", result_path)


if __name__ == "__main__":
    main()
