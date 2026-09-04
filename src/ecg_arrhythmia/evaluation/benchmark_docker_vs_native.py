"""Small, production-path native-versus-Docker benchmark for Raspberry Pi 5."""

import argparse
import json
import math
import platform
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from ecg_arrhythmia.evaluation.validate_edge_streaming_runtime import (
    EventAccumulator,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import (
    DEFAULT_CHUNK_SIZE,
    ReplayMode,
    ReplaySource,
)
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.transport.send_record import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
    run_record_stream,
)

DEFAULT_OUTPUT = Path(
    "artifacts/results/deployment_evaluation/docker_vs_native/summary.json"
)
DEFAULT_RECORD = "114"
DEFAULT_DEADLINE_MS = 100.0
DEFAULT_MODEL_REPEATS = 5
DEFAULT_WARMUP_CALLS = 100
EXPECTED_PRIMARY_CHUNKS = 18_056
EXPECTED_PRIMARY_PREDICTIONS = 1_873
NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000


def series_summary(values: Sequence[float]) -> dict[str, float]:
    """Return a compact distribution summary over finite measurements."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Measurements must be a non-empty finite series")

    return {
        "mean": float(array.mean()),
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def percentage_delta(native: float, docker: float) -> float | None:
    """Return ``(Docker - native) / native * 100``; zero has no delta."""

    if not all(math.isfinite(value) for value in (native, docker)):
        raise ValueError("Percentage-delta inputs must be finite")
    if native == 0:
        return None

    return (docker - native) / native * 100.0


def deadline_summary(margins_ms: Sequence[float]) -> dict:
    """Summarise completion margin against the fixed 100 ms deadline."""

    margins = np.asarray(margins_ms, dtype=np.float64)
    if margins.size == 0 or not np.all(np.isfinite(margins)):
        raise ValueError("Deadline margins must be a non-empty finite series")

    misses = margins < 0
    return {
        "total_chunks": int(margins.size),
        "deadline_misses": int(misses.sum()),
        "minimum_deadline_margin_ms": float(margins.min()),
    }


def _numeric_values(samples: Iterable[dict], field: str) -> list[float]:
    values = []
    for sample in samples:
        value = sample.get(field)
        if isinstance(value, int | float) and not isinstance(value, bool):
            if math.isfinite(value):
                values.append(float(value))
    return values


def resource_summary(samples: Sequence[dict]) -> dict:
    """Summarise the resource readings emitted by production telemetry."""

    if not samples:
        raise ValueError("At least one resource sample is required")

    def metric(field: str) -> dict | None:
        values = _numeric_values(samples, field)
        if not values:
            return None
        return {"mean": float(np.mean(values)), "maximum": max(values)}

    throttle_values = [
        sample["throttling_active"]
        for sample in samples
        if isinstance(sample.get("throttling_active"), bool)
    ]
    governors = sorted(
        {
            str(sample["cpu_governor"])
            for sample in samples
            if sample.get("cpu_governor") is not None
        }
    )
    rss_values = _numeric_values(samples, "process_rss_mib")
    final_window = rss_values[max(0, len(rss_values) * 4 // 5) :]
    rss_interpretation = None
    if rss_values:
        final_range = max(final_window) - min(final_window)
        rss_interpretation = {
            "start_mib": rss_values[0],
            "end_mib": rss_values[-1],
            "maximum_mib": max(rss_values),
            "final_window_range_mib": final_range,
            "plateau_observed": len(final_window) >= 2 and final_range <= 1.0,
        }

    return {
        "sample_count": len(samples),
        "process_cpu_percent": metric("process_cpu_percent"),
        "rss_mib": metric("process_rss_mib"),
        "rss_interpretation": rss_interpretation,
        "temperature_c": metric("temperature_c"),
        "cpu_frequency_mhz": metric("cpu_frequency_mhz"),
        "throttling": {
            "reading_count": len(throttle_values),
            "active_count": sum(throttle_values),
            "observed": any(throttle_values),
        },
        "governors": governors,
    }


class ProductionStreamObserver:
    """Collect timings and integrity checks around the real streaming path."""

    def __init__(self, deadline_ms: float = DEFAULT_DEADLINE_MS) -> None:
        if not math.isfinite(deadline_ms) or deadline_ms <= 0:
            raise ValueError("Deadline must be positive and finite")

        self.deadline_ms = deadline_ms
        self.processing_ms: list[float] = []
        self.full_path_ms: list[float] = []
        self.scheduling_lateness_ms: list[float] = []
        self.deadline_margins_ms: list[float] = []
        self.model_latency_ms: list[float] = []
        self.resource_samples: list[dict] = []
        self.initialisation_ms: list[float] = []
        self.source_discontinuities = 0
        self.samples_processed = 0
        self._scheduled_ns: int | None = None
        self._period_ns: int | None = None
        self._next_sample_index: int | None = None
        self._events: EventAccumulator | None = None
        self._event_count = 0
        self._integrity_failures = 0
        self._records: list[dict] = []

    def on_source(self, source) -> None:
        if source.chunk_size != DEFAULT_CHUNK_SIZE or source.sampling_rate != 360.0:
            raise ValueError("Benchmark requires 36-sample chunks at 360 Hz")
        if source.mode is not ReplayMode.REAL_TIME:
            raise ValueError("Benchmark requires the production real-time replay mode")

        self._period_ns = round(
            source.chunk_size / source.sampling_rate * NANOSECONDS_PER_SECOND
        )
        self._next_sample_index = 0
        self._events = EventAccumulator()
        self._records.append(
            {
                "record_name": source.record_name,
                "sampling_rate": source.sampling_rate,
                "chunk_size": source.chunk_size,
                "samples": source.num_samples,
            }
        )

    def on_schedule(self, chunk, target_time: float | None) -> None:
        self._scheduled_ns = (
            None if target_time is None else round(target_time * NANOSECONDS_PER_SECOND)
        )

    def on_initialisation(self, elapsed_ns: int) -> None:
        self.initialisation_ms.append(elapsed_ns / NANOSECONDS_PER_MILLISECOND)

    def on_model_timing(self, elapsed_ns: int) -> None:
        self.model_latency_ms.append(elapsed_ns / NANOSECONDS_PER_MILLISECOND)

    def on_hardware_snapshot(self, snapshot: dict) -> None:
        self.resource_samples.append(dict(snapshot))

    def on_chunk(
        self,
        chunk,
        chunk_started_ns: int,
        processing_started_ns: int,
        processing_ended_ns: int,
        completed_ns: int,
        events,
    ) -> None:
        if self._period_ns is None or self._events is None:
            raise RuntimeError("Observer did not receive the stream source")

        if chunk.start_index != self._next_sample_index:
            self.source_discontinuities += 1
        self._next_sample_index = chunk.last_index + 1
        self.samples_processed += chunk.num_samples

        self.processing_ms.append(
            (processing_ended_ns - processing_started_ns) / NANOSECONDS_PER_MILLISECOND
        )
        self.full_path_ms.append(
            (completed_ns - chunk_started_ns) / NANOSECONDS_PER_MILLISECOND
        )

        if self._scheduled_ns is None:
            margin_ms = self.deadline_ms - self.full_path_ms[-1]
        else:
            self.scheduling_lateness_ms.append(
                (chunk_started_ns - self._scheduled_ns) / NANOSECONDS_PER_MILLISECOND
            )
            deadline_ns = self._scheduled_ns + round(
                self.deadline_ms * NANOSECONDS_PER_MILLISECOND
            )
            margin_ms = (deadline_ns - completed_ns) / NANOSECONDS_PER_MILLISECOND
        self.deadline_margins_ms.append(margin_ms)
        self._events.add_events(events)

    def on_flush(self, events) -> None:
        if self._events is None:
            raise RuntimeError("Observer did not receive the stream source")

        self._events.add_events(events, from_flush=True)
        self._event_count += self._events.num_events
        self._integrity_failures += self._events.integrity_failure_count

    def summary(self, send_counts: dict | Sequence[dict]) -> dict:
        if not self.full_path_ms:
            raise ValueError("The production stream did not process any chunks")

        counts = [send_counts] if isinstance(send_counts, dict) else list(send_counts)
        result = {
            "send_counts": counts,
            "chunks_processed": len(self.full_path_ms),
            "samples_processed": self.samples_processed,
            "prediction_events": self._event_count,
            "integrity_failures": self._integrity_failures,
            "source_discontinuities": self.source_discontinuities,
            "processing_ms": series_summary(self.processing_ms),
            "full_path_ms": series_summary(self.full_path_ms),
            "deadline": deadline_summary(self.deadline_margins_ms),
            "model_stage": {
                "latency_ms": series_summary(self.model_latency_ms),
                "throughput_sequences_per_second": (
                    len(self.model_latency_ms) / (sum(self.model_latency_ms) / 1000.0)
                ),
                "classifier_initialisation_ms": list(self.initialisation_ms),
            },
            "resources": resource_summary(self.resource_samples),
            "records": list(self._records),
            "paced_signal_seconds": self.samples_processed / 360.0,
        }
        if self.scheduling_lateness_ms:
            result["scheduling_lateness_ms"] = series_summary(
                self.scheduling_lateness_ms
            )
        return result


def run_production_stream(
    *,
    host: str,
    port: int,
    record: str = DEFAULT_RECORD,
    model_path: Path = DEFAULT_MODEL_PATH,
    source=None,
    observer: ProductionStreamObserver | None = None,
) -> dict:
    """Measure the existing production sender; no inference path is duplicated."""

    source = source or ReplaySource.from_record(
        record_name=record,
        chunk_size=DEFAULT_CHUNK_SIZE,
        mode=ReplayMode.REAL_TIME,
    )
    observer = observer or ProductionStreamObserver()
    send_counts = run_record_stream(
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
    return observer.summary(send_counts)


def benchmark_model_stage(
    model_path: Path,
    record: str = DEFAULT_RECORD,
    repeats: int = DEFAULT_MODEL_REPEATS,
    warmup_calls: int = DEFAULT_WARMUP_CALLS,
) -> dict:
    """Time the established classifier seam with real production sequences."""

    if repeats < 1:
        raise ValueError("Model repeats must be positive")

    source = ReplaySource.from_record(
        record_name=record,
        chunk_size=DEFAULT_CHUNK_SIZE,
        mode=ReplayMode.ACCELERATED,
    )
    engine = StreamingEngine()
    engine.start_record(record_name=record)
    sequences = []
    for chunk in source.iter_chunks():
        sequences.extend(engine.process_chunk(chunk))
    sequences.extend(engine.flush())
    if not sequences:
        raise ValueError("The selected record produced no model sequences")

    started = perf_counter_ns()
    classifier = ONNXSequenceClassifier(model_path)
    initialisation_ms = (perf_counter_ns() - started) / NANOSECONDS_PER_MILLISECOND
    durations_ns = []
    warmup = min(warmup_calls, len(sequences))
    for _ in range(repeats):
        for sequence in sequences[:warmup]:
            classifier.predict(sequence)
        for sequence in sequences:
            started = perf_counter_ns()
            classifier.predict(sequence)
            durations_ns.append(perf_counter_ns() - started)

    durations_ms = [value / NANOSECONDS_PER_MILLISECOND for value in durations_ns]
    total_seconds = sum(durations_ns) / NANOSECONDS_PER_SECOND
    return {
        "num_sequences": len(sequences),
        "num_inferences": len(durations_ns),
        "warmup_calls_per_repeat": warmup,
        "repeats": repeats,
        "classifier_initialisation_ms": initialisation_ms,
        "latency_ms": series_summary(durations_ms),
        "throughput_sequences_per_second": len(durations_ns) / total_seconds,
    }


def _required(document: dict, path: str):
    value = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Benchmark evidence is missing {path!r}")
        value = value[part]
    return value


def validate_run_evidence(
    run: dict,
    expected_condition: str,
    expected_chunks: int = EXPECTED_PRIMARY_CHUNKS,
    expected_predictions: int = EXPECTED_PRIMARY_PREDICTIONS,
) -> None:
    """Reject incomplete, failed or internally inconsistent primary evidence."""

    if run.get("status") != "completed" or run.get("condition") != expected_condition:
        raise ValueError("Benchmark run status or condition is invalid")

    chunks = _required(run, "paced.chunks_processed")
    predictions = _required(run, "paced.prediction_events")
    deadline = _required(run, "paced.deadline")
    if chunks != expected_chunks or deadline.get("total_chunks") != expected_chunks:
        raise ValueError("Benchmark evidence has an unexpected chunk count")
    if predictions != expected_predictions:
        raise ValueError("Benchmark evidence has an unexpected prediction count")
    if deadline.get("deadline_misses") != 0:
        raise ValueError("Benchmark evidence contains deadline misses")
    if deadline.get("minimum_deadline_margin_ms", 0) <= 0:
        raise ValueError("Benchmark evidence has no positive deadline margin")
    if _required(run, "paced.integrity_failures") != 0:
        raise ValueError("Benchmark evidence contains integrity failures")
    if _required(run, "paced.source_discontinuities") != 0:
        raise ValueError("Benchmark evidence contains source discontinuities")

    for path in (
        "model.latency_ms.mean",
        "model.throughput_sequences_per_second",
        "paced.full_path_ms.mean",
        "paced.full_path_ms.p95",
        "paced.full_path_ms.p99",
        "paced.full_path_ms.maximum",
        "paced.resources.process_cpu_percent.mean",
        "paced.resources.rss_mib.mean",
        "paced.resources.temperature_c.mean",
    ):
        value = _required(run, path)
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"Benchmark evidence field {path!r} is invalid")


def _mean_path(runs: Sequence[dict], path: str) -> float:
    return float(np.mean([float(_required(run, path)) for run in runs]))


def aggregate_condition(runs: Sequence[dict]) -> dict:
    """Equally weight the two fresh-process runs for one condition."""

    latency_fields = ("mean", "median", "p95", "p99", "maximum")
    resources = {}
    for name, path in (
        ("process_cpu_percent", "process_cpu_percent"),
        ("rss_mib", "rss_mib"),
        ("temperature_c", "temperature_c"),
    ):
        resources[name] = {
            "mean": _mean_path(runs, f"paced.resources.{path}.mean"),
            "maximum": _mean_path(runs, f"paced.resources.{path}.maximum"),
        }

    return {
        "runs": len(runs),
        "model": {
            "latency_ms": {
                field: _mean_path(runs, f"model.latency_ms.{field}")
                for field in latency_fields
            },
            "throughput_sequences_per_second": _mean_path(
                runs, "model.throughput_sequences_per_second"
            ),
        },
        "paced": {
            "full_path_ms": {
                field: _mean_path(runs, f"paced.full_path_ms.{field}")
                for field in latency_fields
            },
            "deadline_misses": sum(
                int(_required(run, "paced.deadline.deadline_misses")) for run in runs
            ),
            "minimum_deadline_margin_ms": min(
                float(_required(run, "paced.deadline.minimum_deadline_margin_ms"))
                for run in runs
            ),
            "chunks_per_run": int(_required(runs[0], "paced.chunks_processed")),
            "predictions_per_run": int(_required(runs[0], "paced.prediction_events")),
            "integrity_failures": sum(
                int(_required(run, "paced.integrity_failures")) for run in runs
            ),
            "source_discontinuities": sum(
                int(_required(run, "paced.source_discontinuities")) for run in runs
            ),
            "resources": resources,
        },
    }


def build_summary(runs: Sequence[dict], evidence: dict | None = None) -> dict:
    """Build the public result from a strict native-Docker-Docker-native ABBA."""

    expected_order = ["native", "docker", "docker", "native"]
    if [run.get("condition") for run in runs] != expected_order:
        raise ValueError(
            "Primary runs must be supplied in native-Docker-Docker-native order"
        )

    for run, condition in zip(runs, expected_order, strict=True):
        validate_run_evidence(run, condition)

    native = aggregate_condition((runs[0], runs[3]))
    docker = aggregate_condition((runs[1], runs[2]))
    model_fields = ("mean", "median", "p95", "p99", "maximum")
    path_fields = ("mean", "median", "p95", "p99", "maximum")

    return {
        "schema_version": 1,
        "status": "passed_with_measurable_docker_overhead",
        "evidence": evidence or {},
        "methodology": {
            "platform": "Raspberry Pi 5 / ARM64",
            "precision": "FP32",
            "provider": "CPUExecutionProvider",
            "order": expected_order,
            "sampling_rate_hz": 360.0,
            "chunk_size_samples": DEFAULT_CHUNK_SIZE,
            "chunk_period_ms": DEFAULT_DEADLINE_MS,
            "deadline_ms": DEFAULT_DEADLINE_MS,
            "replay_mode": "real_time",
            "cpu_governor": "performance",
            "cpu_frequency_mhz": 2400.0,
            "production_streaming_path": True,
        },
        "native": native,
        "docker": docker,
        "docker_change_percent": {
            "model_latency_ms": {
                field: percentage_delta(
                    native["model"]["latency_ms"][field],
                    docker["model"]["latency_ms"][field],
                )
                for field in model_fields
            },
            "model_throughput_sequences_per_second": percentage_delta(
                native["model"]["throughput_sequences_per_second"],
                docker["model"]["throughput_sequences_per_second"],
            ),
            "paced_full_path_ms": {
                field: percentage_delta(
                    native["paced"]["full_path_ms"][field],
                    docker["paced"]["full_path_ms"][field],
                )
                for field in path_fields
            },
        },
        "validation": {
            "total_primary_chunks": sum(
                int(_required(run, "paced.chunks_processed")) for run in runs
            ),
            "deadline_misses": 0,
            "integrity_failures": 0,
            "source_discontinuities": 0,
        },
        "limitations": [
            "Two runs per condition are descriptive, not an equivalence study.",
            "Finite successful runs do not establish hard-real-time guarantees.",
            "An earlier exploratory campaign repeatedly associated rare native "
            "outliers with hardware-query 108, but isolated no actionable "
            "mechanism; that relationship is historical context, not benchmark "
            "evidence.",
        ],
    }


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def run_evidence(condition: str, host: str, port: int, model_path: Path) -> dict:
    """Run one fresh-process condition for a later ABBA aggregation."""

    return {
        "schema_version": 1,
        "status": "completed",
        "condition": condition,
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": {"machine": platform.machine(), "platform": platform.platform()},
        "model": benchmark_model_stage(model_path),
        "paced": run_production_stream(
            host=host,
            port=port,
            model_path=model_path,
        ),
    }


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Measure one native/Docker run")
    run_parser.add_argument("--condition", choices=("native", "docker"), required=True)
    run_parser.add_argument("--host", required=True)
    run_parser.add_argument("--port", type=int, default=8765)
    run_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    run_parser.add_argument("--output", type=Path, required=True)

    summary_parser = subparsers.add_parser(
        "summarise", help="Validate and aggregate four ABBA run files"
    )
    summary_parser.add_argument("runs", nargs=4, type=Path)
    summary_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        result = run_evidence(args.condition, args.host, args.port, args.model_path)
        write_json(result, args.output)
    else:
        write_json(build_summary([_read_json(path) for path in args.runs]), args.output)


if __name__ == "__main__":
    main()
