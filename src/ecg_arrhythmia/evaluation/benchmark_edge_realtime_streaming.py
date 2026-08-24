import argparse
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from pathlib import Path
from time import perf_counter_ns, sleep

import numpy as np

from ecg_arrhythmia.evaluation.benchmark_edge_quantized_inference import (
    benchmark_health_snapshot,
    timing_interpretation_warning,
)
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import (
    environment_metadata,
    latency_summary_ms,
)
from ecg_arrhythmia.evaluation.validate_edge_streaming_runtime import EventAccumulator
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplaySource
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.streaming.streaming_predictor import StreamingPredictor

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_FP32_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_INT8_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer_int8.onnx")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/deployment_evaluation/edge_realtime_streaming"
)

NANOSECONDS_PER_SECOND = 1_000_000_000
NANOSECONDS_PER_MILLISECOND = 1_000_000
MAX_REPORTED_MISS_INDICES = 10

Clock = Callable[[], int]
Sleep = Callable[[float], None]


def chunk_period_ns(chunk_size: int, sampling_rate: float) -> int:
    """Return the signal duration represented by one chunk in nanoseconds."""

    if chunk_size < 1 or sampling_rate <= 0:
        raise ValueError("Chunk size and sampling rate must be positive.")

    return round(chunk_size / sampling_rate * NANOSECONDS_PER_SECOND)


def run_paced(
    predictor: StreamingPredictor,
    chunks: Iterable[SampleChunk],
    period_ns: int,
    clock: Clock = perf_counter_ns,
    sleeper: Sleep = sleep,
) -> dict:
    """Replay chunks according to their real-time arrival schedule."""

    # Create an event accumulator for this record. We will use this
    # to extract some statistics later.
    accumulator = EventAccumulator()

    scheduled_times: list[int] = []
    start_times: list[int] = []
    completion_times: list[int] = []
    processing_times: list[int] = []

    # Starts the clock for this record
    replay_start = clock()

    # For each yeilded chunk
    for chunk_index, chunk in enumerate(chunks):
        # Calculate when this chunk should arrive
        scheduled_time = replay_start + chunk_index * period_ns

        # Get the current time.
        current_time = clock()

        # If the current time is less than the scheduled time, then
        # this chunk has arrived too quick, so we should wait.
        # sleeper is in seconds, so we have to convert to seconds.
        if current_time < scheduled_time:
            wait_seconds = (scheduled_time - current_time) / NANOSECONDS_PER_SECOND

            sleeper(wait_seconds)

        # Process the chunk and compute how long it takes.
        processing_start = clock()
        events = predictor.process_chunk(chunk)
        processing_end = clock()

        # Record timing information
        scheduled_times.append(scheduled_time)
        start_times.append(processing_start)
        completion_times.append(processing_end)
        processing_times.append(processing_end - processing_start)

        accumulator.add_events(events)

    # Measure total wall-clock time for the paced replay
    paced_wall_ns = clock() - replay_start

    # Flush any remaining predictions
    flush_start = clock()
    flush_events = predictor.flush()
    flush_ns = clock() - flush_start

    # Add the remaining flush events to the accumulator
    accumulator.add_events(flush_events, from_flush=True)

    return {
        "scheduled_ns": scheduled_times,
        "actual_start_ns": start_times,
        "completion_ns": completion_times,
        "processing_ns": processing_times,
        "paced_wall_ns": paced_wall_ns,
        "flush_ns": flush_ns,
        "accumulator": accumulator,
    }


def scheduling_statistics(
    scheduled_ns: list[int],
    actual_start_ns: list[int],
) -> dict:
    """Summarise how late processing started relative to the ideal schedule."""

    # For times where we were ahead of schedule and had to sleep,
    # processing start is approximatley equal to the scheduled time.
    # When we are behind schedule, we may start processing later than scheduled
    # in which case their difference will be greater. We aim for as close to
    # 0 as possbile.
    lateness_ms = (
        np.asarray(actual_start_ns, dtype=np.int64)
        - np.asarray(scheduled_ns, dtype=np.int64)
    ) / NANOSECONDS_PER_MILLISECOND

    return {
        "mean": float(lateness_ms.mean()),
        "median": float(np.percentile(lateness_ms, 50)),
        "p95": float(np.percentile(lateness_ms, 95)),
        "maximum": float(lateness_ms.max()),
        "final": float(lateness_ms[-1]),
    }


def deadline_statistics(
    scheduled_ns: list[int],
    completion_ns: list[int],
    period_ns: int,
) -> dict:
    """Summarise whether each chunk completed before the next was due."""

    scheduled = np.asarray(scheduled_ns, dtype=np.int64)
    completion = np.asarray(completion_ns, dtype=np.int64)
    # Processing is late if by the time a chunk has finished processing (completion)
    # we're already scheduled for a new chunk to be processed (scheduled + period_ns)
    lateness_ns = completion - (scheduled + period_ns)
    # If that difference is positive, that means we finished processing a chunk
    # after a new chunk was already due to be processing, so that deadline was
    # missed.
    missed = lateness_ns > 0

    # Total chunks
    total = int(scheduled.size)
    # Total missed deadlines
    misses = int(missed.sum())
    # Convert the lateness values to milliseconds.
    lateness_ms = lateness_ns / NANOSECONDS_PER_MILLISECOND

    return {
        "total_chunks": total,
        "deadline_misses": misses,
        "deadline_miss_percentage": misses / total * 100.0,
        "maximum_deadline_lateness_ms": float(lateness_ms.max()),
        "mean_missed_deadline_lateness_ms": (
            float(lateness_ms[missed].mean()) if misses else None
        ),
        "missed_chunk_indices": [
            int(index) for index in np.nonzero(missed)[0][:MAX_REPORTED_MISS_INDICES]
        ],
    }


def processing_fractions(summary_ms: dict, period_ms: float) -> dict:
    """Return the fraction of each chunk period consumed by processing."""

    if period_ms <= 0:
        raise ValueError("Chunk period must be positive.")

    return {
        # Fraction of one chunk's real-time window used by average processing time.
        # Values below 1 mean average processing finishes before the next chunk arrives.
        "mean_fraction": summary_ms["mean"] / period_ms,
        # Fraction of one chunk's real-time window used by the p95 processing time.
        # Values above 1 mean some slower chunks exceed the nominal chunk period.
        "p95_fraction": summary_ms["p95"] / period_ms,
    }


def accumulator_summary(accumulator: EventAccumulator) -> dict:
    """Return prediction counts and event-integrity information."""

    return {
        "prediction_events": accumulator.num_events,
        "flush_prediction_events": accumulator.flush_event_count,
        "class_counts": dict(accumulator.class_counts),
        "integrity": {
            "passed": accumulator.integrity_passed,
            "failure_count": accumulator.integrity_failure_count,
            "failures": list(accumulator.integrity_failures),
        },
    }


def _chunks(
    source: ReplaySource,
    max_chunks: int | None,
) -> Iterator[SampleChunk]:
    # Get the generator object in ReplaySource
    chunks = source.iter_chunks()
    # Return chunks up to max_chunks, else return none.
    # islice is essentially Python slicing but for iterators.
    # We give it an iterator object and it can tell it when to stop.
    # So this returns the full iterator that will go through the whole source,
    # or the slice iterator that will go up to max_chunks
    return chunks if max_chunks is None else islice(chunks, max_chunks)


def paced_record_run(
    model_path: Path,
    precision: str,
    record_name: str,
    chunk_size: int,
    max_chunks: int | None = None,
) -> dict:
    """Run one complete real-time paced replay with one model."""

    # Populate Replay Source with the selected record.
    source = ReplaySource.from_record(
        record_name=record_name,
        chunk_size=chunk_size,
    )

    # Calculate how much ECG signal each chunk represents in nanoseconds.
    period_ns = chunk_period_ns(chunk_size, source.sampling_rate)

    # Create the classifier
    classifier = ONNXSequenceClassifier(model_path)

    # Create the predictor
    predictor = StreamingPredictor(
        engine=StreamingEngine(),
        classifier=classifier,
    )
    # Resets the Engine state
    predictor.start_record(record_name=record_name)

    logger.info(
        "Paced %s replay of record %s: %d chunks at %.3f ms per chunk",
        precision.upper(),
        record_name,
        source.num_chunks if max_chunks is None else max_chunks,
        period_ns / NANOSECONDS_PER_MILLISECOND,
    )

    # Returns the paced-run statistics
    run = run_paced(
        predictor=predictor,
        chunks=_chunks(source, max_chunks),
        period_ns=period_ns,
    )

    # Calculates the min, max, mean, median, and p95 of the distribution of
    # processing each chunk
    summary_ms = latency_summary_ms(run["processing_ns"])

    # Calculate how much ECG signal each chunk represents in milliseconds.
    period_ms = period_ns / NANOSECONDS_PER_MILLISECOND

    return {
        "precision": precision,
        "model_path": str(model_path),
        "record_name": record_name,
        "sampling_rate": source.sampling_rate,
        "chunk_size": chunk_size,
        "chunk_period_ms": period_ms,
        "num_samples": source.num_samples,
        "signal_duration_seconds": source.num_samples / source.sampling_rate,
        "truncated": max_chunks is not None,
        "chunks_processed": len(run["processing_ns"]),
        "paced_wall_seconds": run["paced_wall_ns"] / NANOSECONDS_PER_SECOND,
        "chunk_processing_ms": summary_ms,
        "deadline_utilisation": processing_fractions(summary_ms, period_ms),
        "scheduling_lateness_ms": scheduling_statistics(
            run["scheduled_ns"],
            run["actual_start_ns"],
        ),
        "deadline": deadline_statistics(
            run["scheduled_ns"],
            run["completion_ns"],
            period_ns,
        ),
        "flush_ms": run["flush_ns"] / NANOSECONDS_PER_MILLISECOND,
        **accumulator_summary(run["accumulator"]),
        "provider": classifier.providers[0],
        "raw_arrays": {
            "scheduled_ns": np.asarray(run["scheduled_ns"], dtype=np.int64),
            "actual_start_ns": np.asarray(run["actual_start_ns"], dtype=np.int64),
            "completion_ns": np.asarray(run["completion_ns"], dtype=np.int64),
            "processing_ns": np.asarray(run["processing_ns"], dtype=np.int64),
        },
    }


def _runtime_validation(summary: dict) -> dict:
    """Return a correctness verdict without inventing a timing threshold."""

    reasons = []

    if summary["prediction_events"] == 0:
        reasons.append("no PredictionEvents were emitted")

    if not summary["integrity"]["passed"]:
        reasons.append(
            f"{summary['integrity']['failure_count']} event integrity checks failed"
        )

    return {
        "status": "PASSED" if not reasons else "FAILED",
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the complete production streaming pipeline on the "
            "Raspberry Pi at the signal's real arrival rate."
        )
    )
    parser.add_argument("--record-name", type=str, default=DEFAULT_RECORD_NAME)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--precision-label",
        choices=("fp32", "int8"),
        default="fp32",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Defaults to the model matching --precision-label.",
    )
    parser.add_argument(
        "--paced-max-chunks",
        type=int,
        default=None,
        help="Development smoke only; marks the result as truncated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _write_outputs(
    result: dict,
    raw_arrays: dict[str, np.ndarray],
    output_dir: Path,
    stem: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    np.savez_compressed(output_dir / f"{stem}_raw.npz", **raw_arrays)
    return json_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Extract CL arguments
    args = parse_args()

    default_models = {
        "fp32": DEFAULT_FP32_MODEL_PATH,
        "int8": DEFAULT_INT8_MODEL_PATH,
    }
    model_path = args.model_path or default_models[args.precision_label]

    # Creates a snapshot of the pis total ram, available ram, throttle flags
    # CPU frequency, and CPU governour before benchmarking.
    health_before = benchmark_health_snapshot()

    # Returns statistics about the paced run
    result = paced_record_run(
        model_path=model_path,
        precision=args.precision_label,
        record_name=args.record_name,
        chunk_size=args.chunk_size,
        max_chunks=args.paced_max_chunks,
    )

    # Creates a snapshot of the pis total ram, available ram, throttle flags
    # CPU frequency, and CPU governour after benchmarking.
    health_after = benchmark_health_snapshot()

    # Extract the raw arrays dict from result. This also removes raw_arrays
    # from result
    raw_arrays = result.pop("raw_arrays")

    # Add hardware health to our result dict
    result["hardware_health"] = {
        "before": health_before,
        "after": health_after,
    }
    # If there was any throttling metrics should be interpreted with caution
    result["timing_interpretation_warning"] = timing_interpretation_warning(
        health_before["throttled"],
        health_after["throttled"],
    )
    # Returns a pass or fail if runtime has run successfully with no integrity
    # erros
    result["runtime_validation"] = _runtime_validation(result)
    # Returns environmment metadata
    result["environment"] = environment_metadata((result["provider"],))

    stem = f"record_{args.record_name}_{args.precision_label}_paced"
    if args.paced_max_chunks is not None:
        stem += "_truncated"

    json_path = _write_outputs(result, raw_arrays, args.output_dir, stem)
    logger.info("Wrote paced result to %s", json_path)


if __name__ == "__main__":
    main()
