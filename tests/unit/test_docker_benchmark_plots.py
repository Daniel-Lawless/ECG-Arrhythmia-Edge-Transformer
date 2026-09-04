import json

import pytest

from ecg_arrhythmia.evaluation.docker_benchmark_plots import (
    plot_paced_latency,
    plot_sustained_resources,
    write_docker_benchmark_figures,
)


@pytest.fixture
def docker_summary():
    return {
        "methodology": {"deadline_ms": 100.0},
        "native": {
            "paced": {
                "full_path_ms": {"mean": 12.6, "p95": 64.4, "p99": 66.2},
                "minimum_deadline_margin_ms": 28.5,
            }
        },
        "docker": {
            "paced": {
                "full_path_ms": {"mean": 13.6, "p95": 69.5, "p99": 70.8},
                "minimum_deadline_margin_ms": 21.857,
            }
        },
        "docker_change_percent": {
            "paced_full_path_ms": {"mean": 7.9, "p95": 7.9, "p99": 6.9}
        },
        "validation": {"total_primary_chunks": 72_224, "deadline_misses": 0},
    }


@pytest.fixture
def sustained_result():
    return {
        "duration": {"measured_wall_seconds": 3600.3},
        "metrics": {
            "chunks_processed": 36_000,
            "deadline": {
                "deadline_misses": 0,
                "minimum_deadline_margin_ms": 19.9,
            },
        },
        "resources": {
            "temperature_c": {"mean": 48.44, "maximum": 50.7},
            "cpu_frequency_mhz": {"mean": 2400.0, "maximum": 2400.0},
            "rss_mib": {
                "start": 180.92,
                "mean": 196.91,
                "end": 198.34,
                "maximum": 198.36,
            },
            "process_cpu_percent": {"mean": 23.71, "maximum": 27.36},
            "throttling": {"reading_count": 720, "active_count": 0},
        },
    }


def test_paced_latency_figure_is_written(tmp_path, docker_summary):
    output = tmp_path / "paced.png"

    assert plot_paced_latency(docker_summary, output) == output
    assert output.stat().st_size > 0


def test_sustained_resource_summary_is_written(tmp_path, sustained_result):
    output = tmp_path / "resources.png"

    assert plot_sustained_resources(sustained_result, output) == output
    assert output.stat().st_size > 0


def test_entry_point_reads_only_the_two_curated_json_files(
    tmp_path,
    docker_summary,
    sustained_result,
):
    summary_path = tmp_path / "summary.json"
    sustained_path = tmp_path / "sustained.json"
    summary_path.write_text(json.dumps(docker_summary), encoding="utf-8")
    sustained_path.write_text(json.dumps(sustained_result), encoding="utf-8")

    written = write_docker_benchmark_figures(
        summary_path,
        sustained_path,
        tmp_path / "figures",
    )

    assert [path.name for path in written] == [
        "native_vs_docker_paced_latency.png",
        "docker_sustained_resources.png",
    ]
    assert all(path.is_file() for path in written)
