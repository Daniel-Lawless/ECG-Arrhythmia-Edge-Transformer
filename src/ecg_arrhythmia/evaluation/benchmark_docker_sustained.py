"""Run the focused 60-minute Docker production-stream validation."""

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ecg_arrhythmia.evaluation.benchmark_docker_vs_native import (
    DEFAULT_DEADLINE_MS,
    DEFAULT_MODEL_PATH,
    ProductionStreamObserver,
    _required,
    write_json,
)
from ecg_arrhythmia.streaming.replay_source import (
    DEFAULT_CHUNK_SIZE,
    ReplayMode,
    ReplaySource,
)
from ecg_arrhythmia.transport.send_record import (
    DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
    run_record_stream,
)

DEFAULT_OUTPUT = Path(
    "artifacts/results/deployment_evaluation/docker_vs_native/sustained.json"
)
DEFAULT_DURATION_SECONDS = 60 * 60
DEFAULT_EXPECTED_CHUNKS = 36_000
DEFAULT_EXPECTED_PREDICTIONS = 4_329
RECORD_CYCLE = ("114", "122", "209", "210", "231", "233")


def validate_sustained_result(
    result: dict,
    expected_duration_seconds: float = DEFAULT_DURATION_SECONDS,
    expected_chunks: int = DEFAULT_EXPECTED_CHUNKS,
    expected_predictions: int = DEFAULT_EXPECTED_PREDICTIONS,
) -> None:
    """Reject incomplete or invalid sustained benchmark evidence."""

    if result.get("status") != "completed" or result.get("condition") != "docker":
        raise ValueError("Sustained evidence must be a completed Docker run")

    signal_seconds = _required(result, "duration.signal_seconds")
    measured_seconds = _required(result, "duration.measured_wall_seconds")
    if (
        not isinstance(signal_seconds, int | float)
        or not math.isfinite(signal_seconds)
        or abs(signal_seconds - expected_duration_seconds) > DEFAULT_DEADLINE_MS / 1000
    ):
        raise ValueError("Sustained signal duration is invalid")
    if (
        not isinstance(measured_seconds, int | float)
        or not math.isfinite(measured_seconds)
        or measured_seconds < signal_seconds
    ):
        raise ValueError("Sustained measured duration is invalid")

    chunks = _required(result, "metrics.chunks_processed")
    predictions = _required(result, "metrics.prediction_events")
    deadline = _required(result, "metrics.deadline")
    if chunks != expected_chunks or deadline.get("total_chunks") != expected_chunks:
        raise ValueError("Sustained evidence has an unexpected chunk count")
    if predictions != expected_predictions:
        raise ValueError("Sustained evidence has an unexpected prediction count")
    if deadline.get("deadline_misses") != 0:
        raise ValueError("Sustained evidence contains deadline misses")
    if deadline.get("minimum_deadline_margin_ms", 0) <= 0:
        raise ValueError("Sustained evidence has no positive deadline margin")
    if _required(result, "metrics.integrity_failures") != 0:
        raise ValueError("Sustained evidence contains integrity failures")
    if _required(result, "metrics.source_discontinuities") != 0:
        raise ValueError("Sustained evidence contains source discontinuities")

    for path in (
        "metrics.full_path_ms.mean",
        "metrics.full_path_ms.p95",
        "metrics.full_path_ms.p99",
        "metrics.full_path_ms.maximum",
        "resources.process_cpu_percent.mean",
        "resources.rss_mib.mean",
        "resources.temperature_c.maximum",
    ):
        value = _required(result, path)
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Sustained evidence field {path!r} is invalid")

    resources = result["resources"]
    if resources.get("governors") != ["performance"]:
        raise ValueError("Sustained evidence requires the performance governor")
    frequency = resources.get("cpu_frequency_mhz")
    if not frequency or frequency.get("mean") != 2400.0:
        raise ValueError("Sustained evidence requires a measured 2.4 GHz frequency")
    throttling = resources.get("throttling", {})
    if throttling.get("reading_count", 0) < 1 or throttling.get("active_count") != 0:
        raise ValueError("Sustained evidence is missing clean throttling readings")


def build_sustained_result(
    paced: dict,
    measured_wall_seconds: float,
    requested_seconds: float = DEFAULT_DURATION_SECONDS,
) -> dict:
    """Convert one production-path run into the compact sustained schema."""

    result = {
        "schema_version": 1,
        "status": "completed",
        "condition": "docker",
        "timestamp": datetime.now(UTC).isoformat(),
        "methodology": {
            "platform": "Raspberry Pi 5 / ARM64",
            "precision": "FP32",
            "provider": "CPUExecutionProvider",
            "sampling_rate_hz": 360.0,
            "chunk_size_samples": DEFAULT_CHUNK_SIZE,
            "chunk_period_ms": DEFAULT_DEADLINE_MS,
            "deadline_ms": DEFAULT_DEADLINE_MS,
            "replay_mode": "real_time",
            "cpu_governor": "performance",
            "cpu_frequency_mhz": 2400.0,
            "production_streaming_path": True,
        },
        "duration": {
            "requested_seconds": requested_seconds,
            "signal_seconds": paced["paced_signal_seconds"],
            "measured_wall_seconds": measured_wall_seconds,
        },
        "metrics": {
            key: paced[key]
            for key in (
                "chunks_processed",
                "samples_processed",
                "prediction_events",
                "integrity_failures",
                "source_discontinuities",
                "processing_ms",
                "full_path_ms",
                "deadline",
                "model_stage",
                "records",
            )
        },
        "resources": paced["resources"],
        "validation": {
            "deadline_misses": paced["deadline"]["deadline_misses"],
            "integrity_failures": paced["integrity_failures"],
            "source_discontinuities": paced["source_discontinuities"],
        },
        "limitations": [
            "The finite run demonstrates sustained operation, not indefinitely "
            "bounded memory or a hard-real-time guarantee.",
            "An earlier exploratory campaign repeatedly associated rare native "
            "outliers with hardware-query 108, but isolated no actionable "
            "mechanism; that relationship is retained only as historical context.",
        ],
    }
    return result


def _limited_source(record: str, maximum_chunks: int) -> ReplaySource:
    full_source = ReplaySource.from_record(
        record_name=record,
        chunk_size=DEFAULT_CHUNK_SIZE,
        mode=ReplayMode.REAL_TIME,
    )
    samples = min(full_source.num_samples, maximum_chunks * DEFAULT_CHUNK_SIZE)
    return ReplaySource(
        signal=full_source.signal[:samples],
        sampling_rate=full_source.sampling_rate,
        chunk_size=DEFAULT_CHUNK_SIZE,
        mode=ReplayMode.REAL_TIME,
        record_name=record,
        lead_name=full_source.lead_name,
    )


def run_sustained_stream(
    *,
    host: str,
    port: int,
    model_path: Path = DEFAULT_MODEL_PATH,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    record_cycle: tuple[str, ...] = RECORD_CYCLE,
    observer: ProductionStreamObserver | None = None,
    source_factory=_limited_source,
) -> dict:
    """Cycle records through ``run_record_stream`` for the requested duration."""

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Sustained duration must be positive and finite")
    if not record_cycle:
        raise ValueError("At least one source record is required")

    target_chunks = round(duration_seconds / (DEFAULT_DEADLINE_MS / 1000.0))
    observer = observer or ProductionStreamObserver()
    send_counts = []
    started = perf_counter()
    record_index = 0

    while len(observer.full_path_ms) < target_chunks:
        remaining_chunks = target_chunks - len(observer.full_path_ms)
        record = record_cycle[record_index % len(record_cycle)]
        source = source_factory(record, remaining_chunks)
        send_counts.append(
            run_record_stream(
                host=host,
                port=port,
                record=record,
                model_path=model_path,
                chunk_size=DEFAULT_CHUNK_SIZE,
                mode=ReplayMode.REAL_TIME.value,
                runtime_status_interval=DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
                source=source,
                observer=observer,
            )
        )
        record_index += 1

    paced = observer.summary(send_counts)
    return build_sustained_result(
        paced,
        perf_counter() - started,
        requested_seconds=duration_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sustained_stream(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        duration_seconds=args.duration_seconds,
    )
    validate_sustained_result(
        result,
        expected_duration_seconds=args.duration_seconds,
        expected_chunks=round(args.duration_seconds * 10),
        expected_predictions=(
            DEFAULT_EXPECTED_PREDICTIONS
            if args.duration_seconds == DEFAULT_DURATION_SECONDS
            else result["metrics"]["prediction_events"]
        ),
    )
    write_json(result, args.output)


if __name__ == "__main__":
    main()
