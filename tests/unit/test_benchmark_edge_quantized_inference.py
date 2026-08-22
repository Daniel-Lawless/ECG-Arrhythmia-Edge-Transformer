import json
import subprocess
import sys
from pathlib import Path

import pytest

from ecg_arrhythmia.evaluation import benchmark_onnx_inference as benchmark_module
from ecg_arrhythmia.evaluation.benchmark_edge_quantized_inference import (
    benchmark_records_individually,
    cross_platform_comparison,
    global_repeat_durations,
    load_x86_benchmark,
    model_statistics,
    per_record_summary,
    pooled_durations,
    read_cpu_frequency_khz,
    read_cpu_governor,
    timing_interpretation_warning,
)
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import (
    NANOSECONDS_PER_MILLISECOND,
)
from ecg_arrhythmia.evaluation.benchmark_quantized_inference import benchmark_order

MILLISECOND = NANOSECONDS_PER_MILLISECOND


class RecordingClassifier:
    """Classifier stand-in recording every object it is asked to predict."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_ids: list[int] = []

    def predict(self, sequence):
        self.calls += 1
        self.seen_ids.append(id(sequence))

        return sequence


class TickingTimer:
    """Deterministic clock advancing one millisecond per reading."""

    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        self.now_ns += MILLISECOND

        return self.now_ns


class CountingCollector:
    """Sequence collector that scripts per-record sequence lists."""

    def __init__(self, sequences_by_record: dict[str, list]) -> None:
        self.sequences_by_record = sequences_by_record
        self.calls: list[str] = []

    def __call__(self, record_name: str, chunk_size: int) -> list:
        self.calls.append(record_name)

        return self.sequences_by_record[record_name]


def run_fake_benchmark(monkeypatch, sequences_by_record, warmup_calls=0, repeats=2):
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", TickingTimer())
    classifiers = {"fp32": RecordingClassifier(), "int8": RecordingClassifier()}
    collector = CountingCollector(sequences_by_record)

    record_benchmarks, order = benchmark_records_individually(
        record_names=list(sequences_by_record),
        classifiers=classifiers,
        chunk_size=36,
        warmup_calls=warmup_calls,
        repeats=repeats,
        collect=collector,
    )

    return record_benchmarks, order, classifiers, collector


# ---------------------------------------------------------------------
#                 Memory-Conscious Orchestration
# ---------------------------------------------------------------------


def test_each_record_is_collected_exactly_once(monkeypatch):
    _, _, _, collector = run_fake_benchmark(
        monkeypatch,
        {"114": [object()] * 3, "122": [object()] * 2},
    )

    assert collector.calls == ["114", "122"]


def test_no_combined_all_record_collection_is_required(monkeypatch):
    # Records of unequal length benchmark independently: each duration
    # array has that record's length, never the combined total.
    record_benchmarks, _, _, _ = run_fake_benchmark(
        monkeypatch,
        {"114": [object()] * 3, "122": [object()] * 5},
        repeats=2,
    )

    lengths = {
        benchmark["record_name"]: {
            model: [len(repeat) for repeat in repeats]
            for model, repeats in benchmark["durations"].items()
        }
        for benchmark in record_benchmarks
    }

    assert lengths["114"] == {"fp32": [3, 3], "int8": [3, 3]}
    assert lengths["122"] == {"fp32": [5, 5], "int8": [5, 5]}


def test_both_models_see_the_identical_sequence_objects(monkeypatch):
    _, _, classifiers, _ = run_fake_benchmark(
        monkeypatch,
        {"114": [object()] * 4},
        repeats=3,
    )

    # Same objects, same order, for every repeat of every record.
    assert classifiers["fp32"].seen_ids == classifiers["int8"].seen_ids


def test_classifiers_are_reused_across_records_and_repeats(monkeypatch):
    _, _, classifiers, _ = run_fake_benchmark(
        monkeypatch,
        {"114": [object()] * 3, "122": [object()] * 2},
        warmup_calls=1,
        repeats=2,
    )

    # The two original instances absorbed every call: nothing was rebuilt.
    # Per model: 2 repeats x (1 warmup + n timed) per record.
    expected = 2 * (1 + 3) + 2 * (1 + 2)
    assert classifiers["fp32"].calls == expected
    assert classifiers["int8"].calls == expected


def test_warmup_calls_are_excluded_from_timed_arrays(monkeypatch):
    record_benchmarks, _, classifiers, _ = run_fake_benchmark(
        monkeypatch,
        {"114": [object()] * 4},
        warmup_calls=2,
        repeats=1,
    )

    durations = record_benchmarks[0]["durations"]["fp32"]
    assert [len(repeat) for repeat in durations] == [4]
    # Warm-up happened (6 calls total) but produced no timed entries.
    assert classifiers["fp32"].calls == 6


def test_the_counterbalanced_order_is_used_and_returned(monkeypatch):
    _, order, _, _ = run_fake_benchmark(
        monkeypatch,
        {"114": [object()] * 2},
        repeats=4,
    )

    assert order == benchmark_order(4)


def test_a_record_with_no_sequences_fails_loudly(monkeypatch):
    with pytest.raises(ValueError, match="produced no sequences"):
        run_fake_benchmark(monkeypatch, {"114": []})


# ---------------------------------------------------------------------
#                     Pooling Across Records
# ---------------------------------------------------------------------


def _fake_record(name: str, fp32_ms: list[list[int]], int8_ms: list[list[int]]):
    return {
        "record_name": name,
        "num_sequences": len(fp32_ms[0]),
        "durations": {
            "fp32": [[value * MILLISECOND for value in repeat] for repeat in fp32_ms],
            "int8": [[value * MILLISECOND for value in repeat] for repeat in int8_ms],
        },
    }


def test_global_repeat_durations_concatenate_across_records():
    records = [
        _fake_record("A", [[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        _fake_record("B", [[10], [20]], [[30], [40]]),
    ]

    repeat_zero = global_repeat_durations(records, "fp32", 0)

    assert repeat_zero == [
        1 * MILLISECOND,
        2 * MILLISECOND,
        10 * MILLISECOND,
    ]


def test_aggregate_statistics_pool_raw_timings_not_record_summaries():
    # Record A: four 1 ms timings. Record B: one 9 ms timing.
    # Pooled mean = (4*1 + 9) / 5 = 2.6 ms. Averaging the two record
    # means would wrongly give (1 + 9) / 2 = 5 ms.
    records = [
        _fake_record("A", [[1, 1, 1, 1]], [[1, 1, 1, 1]]),
        _fake_record("B", [[9]], [[9]]),
    ]

    statistics = model_statistics(
        records,
        "fp32",
        repeats=1,
        initialisation_ms=12.5,
    )

    assert statistics["pooled_all_repeats"]["mean"] == pytest.approx(2.6)
    assert statistics["per_repeat"][0]["mean"] == pytest.approx(2.6)
    assert statistics["classifier_initialisation_ms"] == 12.5


def test_pooled_durations_cover_every_repeat_and_record():
    records = [
        _fake_record("A", [[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        _fake_record("B", [[10], [20]], [[30], [40]]),
    ]

    pooled = pooled_durations(records, "int8")

    assert sorted(pooled) == sorted(
        value * MILLISECOND for value in [5, 6, 7, 8, 30, 40]
    )


def test_per_record_summary_reports_both_models_and_comparison():
    record = _fake_record("114", [[2, 2], [2, 2]], [[1, 1], [1, 1]])

    summary = per_record_summary(record)

    assert summary["record_name"] == "114"
    assert summary["num_sequences"] == 2
    assert summary["fp32"]["mean"] == pytest.approx(2.0)
    assert summary["int8"]["mean"] == pytest.approx(1.0)
    # INT8 is twice as fast here, so the speedup exceeds one.
    assert summary["comparison"]["mean_latency"]["speedup"] == pytest.approx(2.0)
    assert summary["comparison"]["p95_latency_speedup"] == pytest.approx(2.0)
    assert summary["comparison"]["throughput_change_percentage"] == pytest.approx(100.0)


# ---------------------------------------------------------------------
#                    Cross-Platform Comparison
# ---------------------------------------------------------------------


def _fake_x86_result() -> dict:
    return {
        "fp32": {"across_repeats": {"mean": {"mean": 0.68}}},
        "int8": {"across_repeats": {"mean": {"mean": 3.66}}},
        "comparison": {"mean_latency": {"speedup": 0.186}},
    }


def test_cross_platform_ratios_are_computed_from_both_environments():
    comparison = cross_platform_comparison(
        _fake_x86_result(),
        pi_fp32_mean_ms=3.4,
        pi_int8_mean_ms=1.83,
        pi_int8_speedup=1.858,
    )

    assert comparison["fp32_pi_over_x86_latency_ratio"] == pytest.approx(5.0)
    assert comparison["int8_pi_over_x86_latency_ratio"] == pytest.approx(0.5)
    assert comparison["x86_int8_vs_fp32_speedup"] == pytest.approx(0.186)
    assert comparison["pi_int8_vs_fp32_speedup"] == pytest.approx(1.858)


def test_a_missing_x86_artifact_degrades_to_none(tmp_path):
    assert load_x86_benchmark(tmp_path / "missing.json") is None
    assert cross_platform_comparison(None, 1.0, 1.0, 1.0) is None


def test_a_malformed_x86_artifact_degrades_to_none(tmp_path):
    malformed = tmp_path / "benchmark.json"
    malformed.write_text(json.dumps({"unexpected": "shape"}))

    loaded = load_x86_benchmark(malformed)

    assert loaded == {"unexpected": "shape"}
    assert cross_platform_comparison(loaded, 1.0, 1.0, 1.0) is None


def test_a_real_x86_artifact_is_loaded(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(_fake_x86_result()))

    loaded = load_x86_benchmark(path)

    assert loaded is not None
    assert cross_platform_comparison(loaded, 3.4, 1.83, 1.858) is not None


# ---------------------------------------------------------------------
#                       Health Context Helpers
# ---------------------------------------------------------------------


def test_clean_throttling_produces_no_warning():
    assert timing_interpretation_warning("0x0", "0x0") is None


def test_unknown_throttling_state_produces_no_warning():
    assert timing_interpretation_warning(None, None) is None


def test_dirty_throttling_flags_are_surfaced():
    warning = timing_interpretation_warning("0x0", "0x50000")

    assert warning is not None
    assert "after=0x50000" in warning
    assert "caution" in warning


def test_cpu_context_reads_real_sysfs_values(tmp_path):
    governor = tmp_path / "scaling_governor"
    governor.write_text("ondemand\n")
    frequency = tmp_path / "scaling_cur_freq"
    frequency.write_text("2400000\n")

    assert read_cpu_governor(governor) == "ondemand"
    assert read_cpu_frequency_khz(frequency) == 2400000


def test_missing_cpu_sysfs_files_degrade_to_none(tmp_path):
    assert read_cpu_governor(tmp_path / "missing") is None
    assert read_cpu_frequency_khz(tmp_path / "missing") is None


# ---------------------------------------------------------------------
#                        Runtime-Light Import
# ---------------------------------------------------------------------


def test_importing_the_edge_benchmark_loads_neither_torch_nor_matplotlib():
    # Subprocess isolation: nothing another test imported can leak into
    # the module table being inspected. Plots are lazily imported, so a
    # default (plot-free) import must not pull matplotlib.
    script = (
        "import sys\n"
        "import ecg_arrhythmia.evaluation.benchmark_edge_quantized_inference\n"
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
