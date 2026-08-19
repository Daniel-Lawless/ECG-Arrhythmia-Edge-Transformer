import pytest

from ecg_arrhythmia.evaluation import benchmark_onnx_inference as benchmark_module
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import (
    NANOSECONDS_PER_MILLISECOND,
)
from ecg_arrhythmia.evaluation.benchmark_quantized_inference import (
    across_repeat_summary,
    benchmark_order,
    comparison_metrics,
    latency_comparison,
    per_record_results,
    repeat_summaries,
    run_repeated_benchmark,
    size_comparison,
    summarise_across_repeats,
    throughput_comparison,
    verify_identical_sequences,
)
from ecg_arrhythmia.evaluation.onnx_benchmark_plots import write_benchmark_figures


class CountingClassifier:
    """Classifier stand-in that records how often it was asked to predict."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, sequence):
        self.calls += 1

        return sequence


class TickingTimer:
    """Deterministic nanosecond clock advancing one millisecond per read."""

    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        self.now_ns += NANOSECONDS_PER_MILLISECOND

        return self.now_ns


class _FakeSequence:
    def __init__(self, target_peak_index: int) -> None:
        self.target_peak_index = target_peak_index


# ---------------------------------------------------------------------
#                       Order And Input Guards
# ---------------------------------------------------------------------


def test_the_benchmark_order_alternates_deterministically():
    assert benchmark_order(4) == [
        ("fp32", "int8"),
        ("int8", "fp32"),
        ("fp32", "int8"),
        ("int8", "fp32"),
    ]


def test_invalid_repeat_counts_are_rejected():
    with pytest.raises(ValueError, match="At least one benchmark repeat"):
        benchmark_order(0)

    with pytest.raises(ValueError, match="At least one benchmark repeat"):
        benchmark_order(-3)


def test_matching_sequence_collections_pass_the_guard():
    sequences = [_FakeSequence(100), _FakeSequence(200)]

    verify_identical_sequences(sequences, list(sequences))


def test_mismatched_sequence_counts_fail_the_guard():
    with pytest.raises(ValueError, match="differ in length"):
        verify_identical_sequences([_FakeSequence(1)], [])


def test_mismatched_sequence_order_fails_the_guard():
    with pytest.raises(ValueError, match="diverge at position 1"):
        verify_identical_sequences(
            [_FakeSequence(100), _FakeSequence(200)],
            [_FakeSequence(100), _FakeSequence(999)],
        )


# ---------------------------------------------------------------------
#                        Repeated Benchmarking
# ---------------------------------------------------------------------


def test_repeats_time_every_sequence_and_exclude_warmup(monkeypatch):
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", TickingTimer())
    classifiers = {"fp32": CountingClassifier(), "int8": CountingClassifier()}
    sequences = list(range(4))

    durations, order = run_repeated_benchmark(
        classifiers=classifiers,
        sequences=sequences,
        warmup_calls=2,
        repeats=3,
    )

    assert order == benchmark_order(3)

    for model_name in ("fp32", "int8"):
        # Three repeats, each with four timed durations.
        assert len(durations[model_name]) == 3
        assert all(len(repeat) == 4 for repeat in durations[model_name])
        # Warm-up calls happened but produced no timed entries.
        assert classifiers[model_name].calls == 3 * (2 + 4)


def test_the_same_classifier_objects_are_reused_across_repeats(monkeypatch):
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", TickingTimer())
    fp32 = CountingClassifier()
    int8 = CountingClassifier()

    run_repeated_benchmark(
        classifiers={"fp32": fp32, "int8": int8},
        sequences=list(range(2)),
        warmup_calls=0,
        repeats=2,
    )

    # All calls landed on the two original instances: nothing was rebuilt.
    assert fp32.calls == 4
    assert int8.calls == 4


def test_repeat_summaries_report_latency_and_throughput():
    # Two repeats of two one-millisecond predictions.
    repeat_durations = [
        [NANOSECONDS_PER_MILLISECOND] * 2,
        [2 * NANOSECONDS_PER_MILLISECOND] * 2,
    ]

    summaries = repeat_summaries(repeat_durations, num_sequences=2)

    assert summaries[0]["mean"] == pytest.approx(1.0)
    assert summaries[0]["throughput"] == pytest.approx(1000.0)
    assert summaries[1]["mean"] == pytest.approx(2.0)
    assert summaries[1]["throughput"] == pytest.approx(500.0)


def test_across_repeat_summaries_report_spread():
    summary = summarise_across_repeats([0.5, 0.7, 0.6])

    assert summary["mean"] == pytest.approx(0.6)
    assert summary["median"] == pytest.approx(0.6)
    assert summary["minimum"] == pytest.approx(0.5)
    assert summary["maximum"] == pytest.approx(0.7)


def test_summarising_no_repeats_is_rejected():
    with pytest.raises(ValueError, match="At least one repeat value"):
        summarise_across_repeats([])


def test_per_repeat_p95_values_are_summarised_not_pooled():
    per_repeat = [
        {
            "mean": 1.0,
            "median": 1.0,
            "p95": 2.0,
            "maximum": 3.0,
            "minimum": 0.5,
            "throughput": 1000.0,
        },
        {
            "mean": 1.2,
            "median": 1.1,
            "p95": 4.0,
            "maximum": 5.0,
            "minimum": 0.6,
            "throughput": 800.0,
        },
    ]

    summary = across_repeat_summary(per_repeat)

    # The p95 entry summarises the per-repeat p95 values themselves.
    assert summary["p95"]["mean"] == pytest.approx(3.0)
    assert summary["p95"]["minimum"] == pytest.approx(2.0)
    assert summary["p95"]["maximum"] == pytest.approx(4.0)
    assert summary["throughput"]["mean"] == pytest.approx(900.0)


# ---------------------------------------------------------------------
#                         Comparison Metrics
# ---------------------------------------------------------------------


def test_a_faster_int8_shows_negative_change_and_speedup_above_one():
    comparison = latency_comparison(0.6, 0.4)

    assert comparison["delta_ms"] == pytest.approx(-0.2)
    assert comparison["change_percentage"] == pytest.approx(-33.3333, rel=1e-4)
    assert comparison["speedup"] == pytest.approx(1.5)


def test_a_slower_int8_shows_positive_change_and_speedup_below_one():
    comparison = latency_comparison(0.5, 0.6)

    assert comparison["delta_ms"] == pytest.approx(0.1)
    assert comparison["change_percentage"] == pytest.approx(20.0)
    assert comparison["speedup"] == pytest.approx(0.8333, rel=1e-3)


def test_equal_latencies_compare_cleanly():
    comparison = latency_comparison(0.5, 0.5)

    assert comparison["delta_ms"] == 0.0
    assert comparison["change_percentage"] == 0.0
    assert comparison["speedup"] == pytest.approx(1.0)


def test_throughput_speedup_is_int8_over_fp32():
    comparison = throughput_comparison(1000.0, 1500.0)

    assert comparison["delta"] == pytest.approx(500.0)
    assert comparison["change_percentage"] == pytest.approx(50.0)
    assert comparison["speedup"] == pytest.approx(1.5)


def test_size_comparison_reports_reduction_and_ratio():
    size = size_comparison(1024 * 1024, 512 * 1024)

    assert size["fp32_mib"] == pytest.approx(1.0)
    assert size["int8_mib"] == pytest.approx(0.5)
    assert size["reduction_mib"] == pytest.approx(0.5)
    assert size["reduction_percentage"] == pytest.approx(50.0)
    assert size["compression_ratio"] == pytest.approx(2.0)


def test_comparison_metrics_use_across_repeat_means():
    fp32 = {
        "mean": {"mean": 0.6},
        "median": {"mean": 0.55},
        "p95": {"mean": 0.8},
        "throughput": {"mean": 1600.0},
    }
    int8 = {
        "mean": {"mean": 0.3},
        "median": {"mean": 0.28},
        "p95": {"mean": 0.4},
        "throughput": {"mean": 3200.0},
    }

    comparison = comparison_metrics(fp32, int8)

    assert comparison["mean_latency"]["speedup"] == pytest.approx(2.0)
    assert comparison["p95_latency"]["speedup"] == pytest.approx(2.0)
    assert comparison["throughput"]["speedup"] == pytest.approx(2.0)


SIZE = size_comparison(1024 * 1024, 512 * 1024)


# ---------------------------------------------------------------------
#                         Per-Record Pooling
# ---------------------------------------------------------------------


def test_per_record_results_pool_slices_across_repeats():
    millisecond = NANOSECONDS_PER_MILLISECOND

    # Two records of unequal size over two repeats. Record A occupies
    # positions 0-1, record B position 2.
    durations = {
        "fp32": [
            [1 * millisecond, 3 * millisecond, 10 * millisecond],
            [1 * millisecond, 3 * millisecond, 10 * millisecond],
        ],
        "int8": [
            [2 * millisecond, 2 * millisecond, 5 * millisecond],
            [2 * millisecond, 2 * millisecond, 5 * millisecond],
        ],
    }
    boundaries = [("A", 0, 2), ("B", 2, 3)]

    results = per_record_results(boundaries, durations)

    record_a, record_b = results
    assert record_a["num_sequences"] == 2
    # Record A pools four FP32 timings: 1, 3, 1 and 3 ms.
    assert record_a["fp32"]["mean"] == pytest.approx(2.0)
    assert record_a["int8"]["mean"] == pytest.approx(2.0)
    assert record_a["comparison"]["mean_latency_speedup"] == pytest.approx(1.0)

    # Record B's slice is independent of record A's larger population.
    assert record_b["fp32"]["mean"] == pytest.approx(10.0)
    assert record_b["int8"]["mean"] == pytest.approx(5.0)
    assert record_b["comparison"]["mean_latency_speedup"] == pytest.approx(2.0)
    # Throughput from pooled counts and pooled time: 2 timings in 10 ms.
    assert record_b["int8"]["throughput"] == pytest.approx(200.0)


# ---------------------------------------------------------------------
#                                Plots
# ---------------------------------------------------------------------


def test_benchmark_figures_render_for_a_tiny_example(tmp_path):
    def summary(scale: float) -> dict:
        return {
            metric: {
                "mean": scale,
                "minimum": scale * 0.9,
                "maximum": scale * 1.1,
            }
            for metric in ("mean", "median", "p95", "throughput")
        }

    per_record = [
        {
            "record_name": name,
            "num_sequences": 10,
            "fp32": {"mean": 0.6},
            "int8": {"mean": 0.4},
        }
        for name in ("114", "122")
    ]

    written = write_benchmark_figures(
        fp32_summary=summary(0.6),
        int8_summary=summary(0.4),
        size=SIZE,
        per_record=per_record,
        figures_dir=tmp_path,
    )

    assert len(written) == 4
    assert all(path.exists() for path in written)
    assert {path.suffix for path in tmp_path.iterdir()} == {".png"}
