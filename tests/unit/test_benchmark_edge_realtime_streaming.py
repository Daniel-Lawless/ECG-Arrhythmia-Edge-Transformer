import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.evaluation.benchmark_edge_realtime_streaming import (
    _runtime_validation,
    chunk_period_ns,
    deadline_statistics,
    processing_fractions,
    run_paced,
    scheduling_statistics,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent

MS = 1_000_000  # nanoseconds per millisecond


def make_event(target_peak: int, label: str = "N") -> PredictionEvent:
    label_to_index = {"N": 0, "S": 1, "V": 2, "F": 3}
    logits = np.zeros(4, dtype=np.float32)
    logits[label_to_index[label]] = 1.0

    return PredictionEvent(
        target_peak_index=target_peak,
        peak_indices=(target_peak - 100, target_peak),
        logits=logits,
        predicted_class_index=label_to_index[label],
        predicted_label=label,
    )


class FakeTime:
    """Timeline advanced only by scripted processing and sleeping."""

    def __init__(self) -> None:
        self.now_ns = 0
        self.sleeps_ns: list[int] = []

    def clock(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        duration_ns = round(seconds * 1_000_000_000)
        self.sleeps_ns.append(duration_ns)
        self.now_ns += duration_ns


class ScriptedPredictor:
    """Predictor stand-in with scripted processing times and events."""

    def __init__(
        self,
        fake_time: FakeTime,
        chunk_script: list[tuple[int, list[PredictionEvent]]],
        flush_ns: int = 0,
        flush_events: list[PredictionEvent] | None = None,
    ) -> None:
        self.fake_time = fake_time
        self.chunk_script = chunk_script
        self.flush_ns = flush_ns
        self.flush_events = flush_events or []
        self.flush_calls = 0
        self._next = 0

    def process_chunk(self, chunk) -> list[PredictionEvent]:
        duration_ns, events = self.chunk_script[self._next]
        self._next += 1
        self.fake_time.now_ns += duration_ns
        return events

    def flush(self) -> list[PredictionEvent]:
        self.flush_calls += 1
        self.fake_time.now_ns += self.flush_ns
        return self.flush_events


def paced_run(
    fake_time: FakeTime,
    script: list[tuple[int, list[PredictionEvent]]],
    period_ns: int = 100 * MS,
    **kwargs,
):
    predictor = ScriptedPredictor(fake_time, script, **kwargs)
    run = run_paced(
        predictor,
        [object()] * len(script),
        period_ns=period_ns,
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
    )
    return run, predictor


# ---------------------------------------------------------------------
#                          Chunk Period
# ---------------------------------------------------------------------


def test_thirty_six_samples_at_360_hz_is_exactly_100_ms():
    assert chunk_period_ns(36, 360.0) == 100 * MS


def test_invalid_chunk_period_inputs_are_rejected():
    with pytest.raises(ValueError):
        chunk_period_ns(0, 360.0)

    with pytest.raises(ValueError):
        chunk_period_ns(36, 0.0)


# ---------------------------------------------------------------------
#                        Absolute Scheduling
# ---------------------------------------------------------------------


def test_scheduled_arrivals_are_anchored_to_the_start_not_chained():
    fake_time = FakeTime()
    run, _ = paced_run(
        fake_time,
        [(30 * MS, []), (30 * MS, []), (30 * MS, []), (30 * MS, [])],
    )

    assert run["scheduled_ns"] == [0, 100 * MS, 200 * MS, 300 * MS]


def test_early_arrival_sleeps_exactly_the_remaining_time():
    fake_time = FakeTime()
    run, _ = paced_run(
        fake_time,
        [(30 * MS, []), (30 * MS, []), (30 * MS, [])],
    )

    assert fake_time.sleeps_ns == [70 * MS, 70 * MS]
    assert run["actual_start_ns"] == [0, 100 * MS, 200 * MS]


def test_a_late_pipeline_never_sleeps_a_negative_duration():
    fake_time = FakeTime()
    run, _ = paced_run(
        fake_time,
        [(150 * MS, []), (150 * MS, []), (150 * MS, [])],
    )

    assert fake_time.sleeps_ns == []

    lateness = scheduling_statistics(
        run["scheduled_ns"],
        run["actual_start_ns"],
    )
    assert lateness["maximum"] == pytest.approx(100.0)
    assert lateness["final"] == pytest.approx(100.0)


def test_varying_processing_times_cause_no_cumulative_drift():
    fake_time = FakeTime()
    run, _ = paced_run(
        fake_time,
        [(90 * MS, []), (10 * MS, []), (50 * MS, [])],
    )

    assert run["scheduled_ns"] == [0, 100 * MS, 200 * MS]
    assert fake_time.sleeps_ns == [10 * MS, 90 * MS]
    assert run["actual_start_ns"] == [0, 100 * MS, 200 * MS]

    lateness = scheduling_statistics(
        run["scheduled_ns"],
        run["actual_start_ns"],
    )
    assert lateness["final"] == pytest.approx(0.0)


# ---------------------------------------------------------------------
#                     Processing Duration Boundary
# ---------------------------------------------------------------------


def test_processing_duration_covers_process_chunk_only():
    fake_time = FakeTime()
    run, _ = paced_run(
        fake_time,
        [(90 * MS, []), (10 * MS, []), (50 * MS, [])],
    )

    assert run["processing_ns"] == [90 * MS, 10 * MS, 50 * MS]


# ---------------------------------------------------------------------
#                     Lateness And Deadlines
# ---------------------------------------------------------------------


def test_scheduling_lateness_is_computed_from_known_timestamps():
    lateness = scheduling_statistics(
        scheduled_ns=[0, 100 * MS, 200 * MS],
        actual_start_ns=[0, 105 * MS, 202 * MS],
    )

    assert lateness["mean"] == pytest.approx(7.0 / 3.0)
    assert lateness["median"] == pytest.approx(2.0)
    assert lateness["maximum"] == pytest.approx(5.0)
    assert lateness["final"] == pytest.approx(2.0)


def test_completion_before_the_deadline_is_not_a_miss():
    deadline = deadline_statistics(
        scheduled_ns=[0],
        completion_ns=[30 * MS],
        period_ns=100 * MS,
    )

    assert deadline["deadline_misses"] == 0
    assert deadline["maximum_deadline_lateness_ms"] == pytest.approx(-70.0)
    assert deadline["mean_missed_deadline_lateness_ms"] is None


def test_completion_exactly_on_the_deadline_is_not_a_miss():
    deadline = deadline_statistics(
        scheduled_ns=[0],
        completion_ns=[100 * MS],
        period_ns=100 * MS,
    )

    assert deadline["deadline_misses"] == 0


def test_completion_after_the_deadline_is_a_miss_with_lateness():
    deadline = deadline_statistics(
        scheduled_ns=[0, 100 * MS],
        completion_ns=[120 * MS, 190 * MS],
        period_ns=100 * MS,
    )

    assert deadline["deadline_misses"] == 1
    assert deadline["deadline_miss_percentage"] == pytest.approx(50.0)
    assert deadline["maximum_deadline_lateness_ms"] == pytest.approx(20.0)
    assert deadline["mean_missed_deadline_lateness_ms"] == pytest.approx(20.0)
    assert deadline["missed_chunk_indices"] == [0]


# ---------------------------------------------------------------------
#                    Derived Real-Time Quantities
# ---------------------------------------------------------------------


def test_processing_fractions_relative_to_the_chunk_period():
    fractions = processing_fractions(
        {"mean": 2.0, "p95": 5.0},
        100.0,
    )

    assert fractions["mean_fraction"] == pytest.approx(0.02)
    assert fractions["p95_fraction"] == pytest.approx(0.05)

    with pytest.raises(ValueError):
        processing_fractions({"mean": 2.0, "p95": 5.0}, 0.0)


# ---------------------------------------------------------------------
#                        Flush Accounting
# ---------------------------------------------------------------------


def test_flush_is_called_once_and_timed_separately():
    fake_time = FakeTime()
    run, predictor = paced_run(
        fake_time,
        [(10 * MS, [make_event(100)])],
        flush_ns=8 * MS,
        flush_events=[make_event(200), make_event(300)],
    )

    assert predictor.flush_calls == 1
    assert run["flush_ns"] == 8 * MS
    assert len(run["processing_ns"]) == 1

    accumulator = run["accumulator"]
    assert accumulator.num_events == 3
    assert accumulator.flush_event_count == 2
    assert accumulator.integrity_passed


# ---------------------------------------------------------------------
#                       Runtime Validation
# ---------------------------------------------------------------------


def test_runtime_validation_passes_with_predictions_and_valid_integrity():
    result = _runtime_validation(
        {
            "prediction_events": 5,
            "integrity": {
                "passed": True,
                "failure_count": 0,
                "failures": [],
            },
        }
    )

    assert result == {"status": "PASSED", "reasons": []}


def test_runtime_validation_reports_missing_predictions_and_integrity_failures():
    result = _runtime_validation(
        {
            "prediction_events": 0,
            "integrity": {
                "passed": False,
                "failure_count": 2,
                "failures": ["failure 1", "failure 2"],
            },
        }
    )

    assert result["status"] == "FAILED"
    assert result["reasons"] == [
        "no PredictionEvents were emitted",
        "2 event integrity checks failed",
    ]


# ---------------------------------------------------------------------
#                        Runtime-Light Import
# ---------------------------------------------------------------------


def test_importing_the_realtime_benchmark_loads_neither_torch_nor_matplotlib():
    script = (
        "import sys\n"
        "import ecg_arrhythmia.evaluation.benchmark_edge_realtime_streaming\n"
        "blocked = [name for name in sys.modules "
        "if name == 'torch' or name.startswith('torch.') "
        "or name == 'matplotlib' or name.startswith('matplotlib.')]\n"
        "assert not blocked, f'forbidden modules imported: {blocked}'\n"
        "print('runtime-light import OK')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 0, completed.stderr
    assert "runtime-light import OK" in completed.stdout
