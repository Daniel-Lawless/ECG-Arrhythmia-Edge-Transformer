import copy
import json
from types import SimpleNamespace

import pytest

import ecg_arrhythmia.evaluation.benchmark_docker_vs_native as benchmark


def _run(condition: str, scale: float = 1.0, margin: float = 30.0) -> dict:
    return {
        "schema_version": 1,
        "status": "completed",
        "condition": condition,
        "model": {
            "num_sequences": benchmark.EXPECTED_PRIMARY_PREDICTIONS,
            "num_inferences": benchmark.EXPECTED_PRIMARY_PREDICTIONS * 5,
            "latency_ms": {
                "mean": 2.0 * scale,
                "median": 1.9 * scale,
                "p95": 2.2 * scale,
                "p99": 2.4 * scale,
                "maximum": 3.0 * scale,
            },
            "throughput_sequences_per_second": 500.0 / scale,
        },
        "paced": {
            "chunks_processed": benchmark.EXPECTED_PRIMARY_CHUNKS,
            "prediction_events": benchmark.EXPECTED_PRIMARY_PREDICTIONS,
            "integrity_failures": 0,
            "source_discontinuities": 0,
            "full_path_ms": {
                "mean": 10.0 * scale,
                "median": 1.0 * scale,
                "p95": 20.0 * scale,
                "p99": 30.0 * scale,
                "maximum": 40.0 * scale,
            },
            "deadline": {
                "total_chunks": benchmark.EXPECTED_PRIMARY_CHUNKS,
                "deadline_misses": 0,
                "minimum_deadline_margin_ms": margin,
            },
            "resources": {
                "process_cpu_percent": {
                    "mean": 20.0 * scale,
                    "maximum": 25.0 * scale,
                },
                "rss_mib": {"mean": 170.0, "maximum": 172.0},
                "temperature_c": {"mean": 48.0, "maximum": 50.0},
            },
        },
    }


def test_percentage_delta_uses_native_as_the_baseline():
    assert benchmark.percentage_delta(10.0, 11.0) == pytest.approx(10.0)
    assert benchmark.percentage_delta(10.0, 9.0) == pytest.approx(-10.0)
    assert benchmark.percentage_delta(0.0, 1.0) is None


def test_deadline_summary_counts_misses_and_preserves_worst_margin():
    summary = benchmark.deadline_summary([25.0, 0.0, -0.25, -4.0])

    assert summary == {
        "total_chunks": 4,
        "deadline_misses": 2,
        "minimum_deadline_margin_ms": -4.0,
    }


def test_resource_summary_covers_cpu_rss_temperature_frequency_and_throttling():
    summary = benchmark.resource_summary(
        [
            {
                "process_cpu_percent": 20.0,
                "process_rss_mib": 170.0,
                "temperature_c": 48.0,
                "cpu_frequency_mhz": 2400.0,
                "cpu_governor": "performance",
                "throttling_active": False,
            },
            {
                "process_cpu_percent": 24.0,
                "process_rss_mib": 172.0,
                "temperature_c": 50.0,
                "cpu_frequency_mhz": 2400.0,
                "cpu_governor": "performance",
                "throttling_active": False,
            },
        ]
    )

    assert summary["process_cpu_percent"] == {"mean": 22.0, "maximum": 24.0}
    assert summary["rss_mib"] == {"mean": 171.0, "maximum": 172.0}
    assert summary["rss_interpretation"] == {
        "start_mib": 170.0,
        "end_mib": 172.0,
        "maximum_mib": 172.0,
        "final_window_range_mib": 0.0,
        "plateau_observed": False,
    }
    assert summary["temperature_c"] == {"mean": 49.0, "maximum": 50.0}
    assert summary["cpu_frequency_mhz"]["mean"] == 2400.0
    assert summary["governors"] == ["performance"]
    assert summary["throttling"] == {
        "reading_count": 2,
        "active_count": 0,
        "observed": False,
    }


def test_native_and_docker_runs_are_aggregated_with_equal_run_weight():
    native = benchmark.aggregate_condition([_run("native"), _run("native", 2.0)])

    assert native["model"]["latency_ms"]["mean"] == pytest.approx(3.0)
    assert native["paced"]["full_path_ms"]["p95"] == pytest.approx(30.0)
    assert native["paced"]["resources"]["process_cpu_percent"]["mean"] == 30.0
    assert native["paced"]["chunks_per_run"] == 18_056


def test_build_summary_calculates_deltas_and_uses_worst_headroom():
    runs = [
        _run("native", margin=31.0),
        _run("docker", 1.1, margin=24.0),
        _run("docker", 1.1, margin=22.0),
        _run("native", margin=29.0),
    ]

    summary = benchmark.build_summary(runs)

    assert summary["status"] == "passed_with_measurable_docker_overhead"
    assert summary["docker_change_percent"]["paced_full_path_ms"]["mean"] == (
        pytest.approx(10.0)
    )
    assert summary["native"]["paced"]["minimum_deadline_margin_ms"] == 29.0
    assert summary["docker"]["paced"]["minimum_deadline_margin_ms"] == 22.0
    assert summary["validation"]["total_primary_chunks"] == 72_224


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunks_processed", 18_055, "chunk count"),
        ("prediction_events", 1_872, "prediction count"),
        ("integrity_failures", 1, "integrity failures"),
        ("source_discontinuities", 1, "source discontinuities"),
    ],
)
def test_validation_rejects_count_and_integrity_failures(field, value, message):
    run = _run("native")
    run["paced"][field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.validate_run_evidence(run, "native")


def test_validation_rejects_deadline_misses_and_incomplete_evidence():
    missed = _run("docker")
    missed["paced"]["deadline"]["deadline_misses"] = 1
    with pytest.raises(ValueError, match="deadline misses"):
        benchmark.validate_run_evidence(missed, "docker")

    incomplete = _run("docker")
    del incomplete["paced"]["resources"]["temperature_c"]
    with pytest.raises(ValueError, match="missing"):
        benchmark.validate_run_evidence(incomplete, "docker")


def test_summary_rejects_any_order_other_than_counterbalanced_abba():
    runs = [_run("native"), _run("docker"), _run("native"), _run("docker")]

    with pytest.raises(ValueError, match="native-Docker-Docker-native"):
        benchmark.build_summary(runs)


def test_json_writer_preserves_required_public_structure(tmp_path):
    runs = [_run("native"), _run("docker"), _run("docker"), _run("native")]
    output = tmp_path / "summary.json"

    benchmark.write_json(benchmark.build_summary(runs), output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert set(payload) >= {
        "status",
        "methodology",
        "native",
        "docker",
        "docker_change_percent",
        "validation",
        "limitations",
    }


def test_primary_runner_reuses_run_record_stream(monkeypatch):
    source = SimpleNamespace(record_name="114")
    observer = SimpleNamespace(summary=lambda counts: {"counts": counts})
    calls = []

    def fake_run_record_stream(**kwargs):
        calls.append(kwargs)
        return {"chunks_sent": 18_056}

    monkeypatch.setattr(benchmark, "run_record_stream", fake_run_record_stream)

    result = benchmark.run_production_stream(
        host="receiver",
        port=8765,
        source=source,
        observer=observer,
    )

    assert result == {"counts": {"chunks_sent": 18_056}}
    assert calls[0]["source"] is source
    assert calls[0]["observer"] is observer
    assert calls[0]["mode"] == "real_time"


def test_invalid_numeric_evidence_is_rejected():
    run = copy.deepcopy(_run("native"))
    run["paced"]["full_path_ms"]["mean"] = float("nan")

    with pytest.raises(ValueError, match="invalid"):
        benchmark.validate_run_evidence(run, "native")
