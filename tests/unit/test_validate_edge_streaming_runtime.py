import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.evaluation.validate_edge_streaming_runtime import (
    EventAccumulator,
    compare_runs,
    health_snapshot,
    overall_status,
    runtime_environment,
    stream_and_accumulate,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
from ecg_arrhythmia.telemetry.edge_sensors import (
    parse_meminfo,
    parse_temperature,
    parse_throttled,
    read_meminfo,
    run_vcgencmd,
)


def make_event(
    target_peak: int,
    label: str = "N",
    class_index: int | None = None,
    logits: np.ndarray | None = None,
) -> PredictionEvent:
    label_to_index = {"N": 0, "S": 1, "V": 2, "F": 3}

    if class_index is None:
        class_index = label_to_index.get(label, 0)

    if logits is None:
        logits = np.zeros(4, dtype=np.float32)
        logits[class_index % 4] = 1.0

    return PredictionEvent(
        target_peak_index=target_peak,
        peak_indices=(target_peak - 400, target_peak - 300, target_peak),
        logits=logits,
        predicted_class_index=class_index,
        predicted_label=label,
    )


class ScriptedPredictor:
    """Predictor stand-in emitting pre-scripted events per chunk."""

    def __init__(
        self,
        per_chunk_events: list[list[PredictionEvent]],
        flush_events: list[PredictionEvent],
    ) -> None:
        self.per_chunk_events = per_chunk_events
        self.flush_events = flush_events
        self.flush_calls = 0
        self._next_chunk = 0

    def process_chunk(self, chunk) -> list[PredictionEvent]:
        events = self.per_chunk_events[self._next_chunk]
        self._next_chunk += 1

        return events

    def flush(self) -> list[PredictionEvent]:
        self.flush_calls += 1

        return self.flush_events


# ---------------------------------------------------------------------
#                          Event Accumulation
# ---------------------------------------------------------------------


def test_events_are_counted_by_class_with_first_and_last_targets():
    accumulator = EventAccumulator()

    accumulator.add_events(
        [
            make_event(100, "N"),
            make_event(200, "S"),
            make_event(300, "N"),
            make_event(400, "V"),
            make_event(500, "F"),
        ]
    )

    assert accumulator.num_events == 5
    assert accumulator.class_counts == {"N": 2, "S": 1, "V": 1, "F": 1}
    assert accumulator.first_target_peak == 100
    assert accumulator.last_target_peak == 500
    assert accumulator.integrity_passed


def test_an_empty_run_reports_no_events():
    accumulator = EventAccumulator()

    assert accumulator.num_events == 0
    assert accumulator.first_target_peak is None
    assert accumulator.last_target_peak is None
    assert accumulator.integrity_passed


def test_non_monotonic_target_peaks_fail_integrity():
    accumulator = EventAccumulator()

    accumulator.add_events([make_event(300, "N"), make_event(200, "N")])

    assert not accumulator.integrity_passed
    assert accumulator.integrity_failure_count == 1
    assert "not strictly after" in accumulator.integrity_failures[0]


def test_a_label_and_index_mismatch_fails_integrity():
    accumulator = EventAccumulator()

    accumulator.add_events([make_event(100, "N", class_index=2)])

    assert not accumulator.integrity_passed
    assert "does not match class index" in accumulator.integrity_failures[0]


def test_non_finite_logits_fail_integrity():
    logits = np.array([1.0, np.nan, 0.0, 0.0], dtype=np.float32)
    accumulator = EventAccumulator()

    accumulator.add_events([make_event(100, "N", logits=logits)])

    assert not accumulator.integrity_passed
    assert "non-finite" in accumulator.integrity_failures[0]


def test_wrong_logit_shape_fails_integrity():
    logits = np.zeros(3, dtype=np.float32)
    accumulator = EventAccumulator()

    accumulator.add_events([make_event(100, "N", logits=logits)])

    assert not accumulator.integrity_passed
    assert "logits shape" in accumulator.integrity_failures[0]


def test_failure_messages_are_capped_but_counting_continues():
    accumulator = EventAccumulator()

    # Ten consecutive non-monotonic events at the same position.
    accumulator.add_events([make_event(100, "N")] * 11)

    assert accumulator.integrity_failure_count == 10
    assert len(accumulator.integrity_failures) == 5


# ---------------------------------------------------------------------
#                           Flush Accounting
# ---------------------------------------------------------------------


def test_streaming_counts_chunks_and_includes_flush_events_exactly_once():
    predictor = ScriptedPredictor(
        per_chunk_events=[
            [],
            [make_event(100, "N")],
            [],
            [make_event(200, "S"), make_event(300, "N")],
        ],
        flush_events=[make_event(400, "V")],
    )

    accumulator, num_chunks = stream_and_accumulate(
        predictor,
        [object()] * 4,
    )

    assert num_chunks == 4
    assert predictor.flush_calls == 1
    assert accumulator.num_events == 4
    assert accumulator.flush_called
    assert accumulator.flush_event_count == 1
    assert accumulator.last_target_peak == 400
    assert accumulator.class_counts == {"N": 2, "S": 1, "V": 1, "F": 0}


def test_flush_is_recorded_even_when_it_returns_no_events():
    predictor = ScriptedPredictor(
        per_chunk_events=[[make_event(100, "N")]],
        flush_events=[],
    )

    accumulator, _ = stream_and_accumulate(predictor, [object()])

    assert accumulator.flush_called
    assert accumulator.flush_event_count == 0


# ---------------------------------------------------------------------
#                     FP32 vs INT8 Comparison
# ---------------------------------------------------------------------


def _accumulator_from(events: list[PredictionEvent]) -> EventAccumulator:
    accumulator = EventAccumulator()
    accumulator.add_events(events)

    return accumulator


def test_identical_runs_agree_completely():
    events = [make_event(100, "N"), make_event(200, "S")]

    comparison = compare_runs(
        _accumulator_from(events),
        _accumulator_from(list(events)),
    )

    assert comparison["target_peaks_identical"]
    assert comparison["class_agreements"] == 2
    assert comparison["class_disagreements"] == 0
    assert comparison["class_agreement_percentage"] == pytest.approx(100.0)


def test_class_disagreements_are_counted_without_failing_the_match():
    fp32 = _accumulator_from([make_event(100, "N"), make_event(200, "S")])
    int8 = _accumulator_from([make_event(100, "N"), make_event(200, "N")])

    comparison = compare_runs(fp32, int8)

    assert comparison["target_peaks_identical"]
    assert comparison["class_agreements"] == 1
    assert comparison["class_disagreements"] == 1
    assert comparison["class_agreement_percentage"] == pytest.approx(50.0)


def test_mismatched_target_peaks_are_reported_without_agreement_stats():
    fp32 = _accumulator_from([make_event(100, "N"), make_event(200, "N")])
    int8 = _accumulator_from([make_event(100, "N"), make_event(999, "N")])

    comparison = compare_runs(fp32, int8)

    assert not comparison["target_peaks_identical"]
    assert comparison["class_agreements"] is None
    assert comparison["class_agreement_percentage"] is None


# ---------------------------------------------------------------------
#                       Hardware Health Helpers
# ---------------------------------------------------------------------


def test_vcgencmd_temperature_output_is_parsed():
    assert parse_temperature("temp=48.2'C") == pytest.approx(48.2)


def test_vcgencmd_throttled_output_is_parsed():
    assert parse_throttled("throttled=0x0") == "0x0"
    assert parse_throttled("throttled=0x50000") == "0x50000"


@pytest.mark.parametrize(
    "output",
    [None, "", "garbage", "temp=", "temp=abc'C", "throttled="],
)
def test_unparseable_health_output_degrades_to_none(output):
    assert parse_temperature(output) is None
    assert parse_throttled(output) is None


def test_a_missing_vcgencmd_binary_degrades_to_none():
    def missing_runner(command):
        raise FileNotFoundError("vcgencmd not found")

    assert run_vcgencmd("measure_temp", runner=missing_runner) is None


def test_a_failing_vcgencmd_call_degrades_to_none():
    def failing_runner(command):
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    assert run_vcgencmd("get_throttled", runner=failing_runner) is None


def test_meminfo_is_parsed_into_mib():
    text = "MemTotal:        1013712 kB\nMemAvailable:     825344 kB\nCached: 1 kB\n"

    values = parse_meminfo(text)

    assert values["total_ram_mib"] == pytest.approx(1013712 / 1024)
    assert values["available_ram_mib"] == pytest.approx(825344 / 1024)


def test_a_missing_meminfo_file_degrades_to_none(tmp_path):
    values = read_meminfo(tmp_path / "does_not_exist")

    assert values == {"total_ram_mib": None, "available_ram_mib": None}


def test_health_snapshot_composes_gracefully_without_pi_hardware(tmp_path):
    snapshot = health_snapshot(
        meminfo_path=tmp_path / "missing_meminfo",
        vcgencmd=lambda argument: None,
    )

    assert snapshot == {
        "total_ram_mib": None,
        "available_ram_mib": None,
        "temperature_c": None,
        "throttled": None,
    }


def test_health_snapshot_reports_real_values_when_available(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1024000 kB\nMemAvailable: 512000 kB\n")

    outputs = {
        "measure_temp": "temp=51.0'C",
        "get_throttled": "throttled=0x0",
    }
    snapshot = health_snapshot(
        meminfo_path=meminfo,
        vcgencmd=lambda argument: outputs[argument],
    )

    assert snapshot["total_ram_mib"] == pytest.approx(1000.0)
    assert snapshot["available_ram_mib"] == pytest.approx(500.0)
    assert snapshot["temperature_c"] == pytest.approx(51.0)
    assert snapshot["throttled"] == "0x0"


# ---------------------------------------------------------------------
#                    Environment And Overall Status
# ---------------------------------------------------------------------


def test_runtime_environment_reports_machine_and_runtime():
    environment = runtime_environment(("CPUExecutionProvider",))

    assert environment["architecture"]
    assert environment["python_version"]
    assert environment["onnxruntime_version"]
    assert environment["provider"] == "CPUExecutionProvider"
    assert environment["execution_providers"] == ["CPUExecutionProvider"]


def _passing_summary(precision: str) -> dict:
    return {
        "precision": precision,
        "prediction_events": 10,
        "flush_called": True,
        "integrity": {"passed": True, "failure_count": 0},
    }


def _matching_comparison() -> dict:
    return {"target_peaks_identical": True}


def test_two_clean_runs_with_matching_targets_pass():
    validation = overall_status(
        _passing_summary("fp32"),
        _passing_summary("int8"),
        _matching_comparison(),
    )

    assert validation == {"status": "PASSED", "reasons": []}


def test_a_run_with_no_events_fails_with_a_reason():
    empty = _passing_summary("fp32")
    empty["prediction_events"] = 0

    validation = overall_status(
        empty,
        _passing_summary("int8"),
        _matching_comparison(),
    )

    assert validation["status"] == "FAILED"
    assert any("no PredictionEvents" in reason for reason in validation["reasons"])


def test_integrity_failures_fail_the_validation():
    broken = _passing_summary("int8")
    broken["integrity"] = {"passed": False, "failure_count": 3}

    validation = overall_status(
        _passing_summary("fp32"),
        broken,
        _matching_comparison(),
    )

    assert validation["status"] == "FAILED"
    assert any("integrity" in reason for reason in validation["reasons"])


def test_mismatched_target_peaks_fail_the_validation():
    validation = overall_status(
        _passing_summary("fp32"),
        _passing_summary("int8"),
        {"target_peaks_identical": False},
    )

    assert validation["status"] == "FAILED"
    assert any("different target peaks" in reason for reason in validation["reasons"])


# ---------------------------------------------------------------------
#                        Runtime-Light Import
# ---------------------------------------------------------------------


def test_importing_the_evaluator_loads_neither_torch_nor_matplotlib():
    # A subprocess keeps this check honest: nothing another test imported
    # can leak into the module table being inspected.
    script = (
        "import sys\n"
        "import ecg_arrhythmia.evaluation.validate_edge_streaming_runtime\n"
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
