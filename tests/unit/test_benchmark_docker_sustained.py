import copy
import json
from types import SimpleNamespace

import pytest

import ecg_arrhythmia.evaluation.benchmark_docker_sustained as sustained
from ecg_arrhythmia.evaluation.benchmark_docker_vs_native import write_json


def _paced() -> dict:
    return {
        "chunks_processed": 36_000,
        "samples_processed": 1_295_984,
        "prediction_events": 4_329,
        "integrity_failures": 0,
        "source_discontinuities": 0,
        "processing_ms": {
            "mean": 13.8,
            "median": 0.03,
            "p95": 70.7,
            "p99": 72.8,
            "maximum": 79.8,
        },
        "full_path_ms": {
            "mean": 13.9,
            "median": 0.1,
            "p95": 70.9,
            "p99": 73.0,
            "maximum": 80.0,
        },
        "deadline": {
            "total_chunks": 36_000,
            "deadline_misses": 0,
            "minimum_deadline_margin_ms": 19.9,
        },
        "model_stage": {
            "latency_ms": {"mean": 1.8},
            "throughput_sequences_per_second": 545.0,
        },
        "records": [{"record_name": "114"}, {"record_name": "122"}],
        "resources": {
            "process_cpu_percent": {"mean": 23.7, "maximum": 27.4},
            "rss_mib": {"mean": 196.9, "maximum": 198.4},
            "rss_interpretation": {"plateau_observed": True},
            "temperature_c": {"mean": 48.4, "maximum": 50.7},
            "cpu_frequency_mhz": {"mean": 2400.0, "maximum": 2400.0},
            "throttling": {
                "reading_count": 720,
                "active_count": 0,
                "observed": False,
            },
            "governors": ["performance"],
        },
        "paced_signal_seconds": 3_599.9555555555557,
    }


def _result() -> dict:
    return sustained.build_sustained_result(_paced(), 3_600.297783223)


def test_valid_60_minute_result_passes_all_hard_checks():
    sustained.validate_sustained_result(_result())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("duration", "signal_seconds"), 3_500.0, "signal duration"),
        (("duration", "measured_wall_seconds"), 3_000.0, "measured duration"),
        (("metrics", "chunks_processed"), 35_999, "chunk count"),
        (("metrics", "prediction_events"), 4_328, "prediction count"),
        (("metrics", "integrity_failures"), 1, "integrity failures"),
        (("metrics", "source_discontinuities"), 1, "source discontinuities"),
    ],
)
def test_duration_counts_and_integrity_failures_are_rejected(path, value, message):
    result = copy.deepcopy(_result())
    result[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        sustained.validate_sustained_result(result)


def test_deadline_misses_and_nonpositive_margin_are_rejected():
    result = _result()
    result["metrics"]["deadline"]["deadline_misses"] = 1
    with pytest.raises(ValueError, match="deadline misses"):
        sustained.validate_sustained_result(result)

    result = _result()
    result["metrics"]["deadline"]["minimum_deadline_margin_ms"] = 0.0
    with pytest.raises(ValueError, match="positive deadline margin"):
        sustained.validate_sustained_result(result)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda result: result["resources"].update(governors=["ondemand"]), "governor"),
        (
            lambda result: result["resources"]["cpu_frequency_mhz"].update(mean=1800.0),
            "2.4 GHz",
        ),
        (
            lambda result: result["resources"]["throttling"].update(active_count=1),
            "throttling",
        ),
    ],
)
def test_governor_frequency_and_throttling_are_validated(mutation, message):
    result = _result()
    mutation(result)

    with pytest.raises(ValueError, match=message):
        sustained.validate_sustained_result(result)


def test_result_structure_retains_resources_and_rss_interpretation(tmp_path):
    output = tmp_path / "sustained.json"
    write_json(_result(), output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert set(payload) >= {
        "status",
        "condition",
        "methodology",
        "duration",
        "metrics",
        "resources",
        "validation",
        "limitations",
    }
    assert payload["resources"]["rss_interpretation"]["plateau_observed"] is True


def test_missing_resource_summary_is_rejected():
    result = _result()
    del result["resources"]["temperature_c"]

    with pytest.raises(ValueError, match="missing"):
        sustained.validate_sustained_result(result)


def test_sustained_runner_reuses_production_streaming_path(monkeypatch):
    calls = []

    class FakeObserver:
        def __init__(self):
            self.full_path_ms = []

        def summary(self, counts):
            paced = _paced()
            paced["chunks_processed"] = 2
            paced["samples_processed"] = 72
            paced["prediction_events"] = 1
            paced["deadline"]["total_chunks"] = 2
            paced["paced_signal_seconds"] = 0.2
            return paced

    observer = FakeObserver()
    source = SimpleNamespace(record_name="114")

    def fake_run_record_stream(**kwargs):
        calls.append(kwargs)
        observer.full_path_ms.extend([5.0, 6.0])
        return {"chunks_sent": 2}

    monkeypatch.setattr(sustained, "run_record_stream", fake_run_record_stream)

    result = sustained.run_sustained_stream(
        host="receiver",
        port=8765,
        duration_seconds=0.2,
        record_cycle=("114",),
        observer=observer,
        source_factory=lambda record, chunks: source,
    )

    assert result["metrics"]["chunks_processed"] == 2
    assert calls[0]["source"] is source
    assert calls[0]["observer"] is observer
    assert calls[0]["mode"] == "real_time"
