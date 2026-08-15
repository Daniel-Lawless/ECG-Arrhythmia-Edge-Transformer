import pytest

from ecg_arrhythmia.evaluation import benchmark_onnx_inference as benchmark_module
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import (
    BYTES_PER_MIB,
    NANOSECONDS_PER_MILLISECOND,
    effective_warmup_calls,
    file_size_mib,
    latency_summary_ms,
    throughput_sequences_per_second,
    time_inference,
    total_seconds,
)

# Ten durations in nanoseconds, one to ten milliseconds.
DURATIONS_NS = [
    milliseconds * NANOSECONDS_PER_MILLISECOND for milliseconds in range(1, 11)
]


class CountingClassifier:
    """Classifier stand-in that records how often it was asked to predict."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, sequence):
        self.calls += 1

        return sequence


class TickingTimer:
    """Deterministic nanosecond clock advancing a fixed step per reading."""

    def __init__(self, step_ns: int = NANOSECONDS_PER_MILLISECOND) -> None:
        self.step_ns = step_ns
        self.now_ns = 0

    def __call__(self) -> int:
        self.now_ns += self.step_ns

        return self.now_ns


# ---------------------------------------------------------------------
#                          Measurement Helpers
# ---------------------------------------------------------------------


def test_file_size_converts_bytes_to_mebibytes():
    assert file_size_mib(BYTES_PER_MIB) == pytest.approx(1.0)
    assert file_size_mib(BYTES_PER_MIB // 2) == pytest.approx(0.5)
    assert file_size_mib(0) == pytest.approx(0.0)


def test_a_negative_file_size_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        file_size_mib(-1)


def test_latency_is_summarised_in_milliseconds():
    summary = latency_summary_ms(DURATIONS_NS)

    assert summary["minimum"] == pytest.approx(1.0)
    assert summary["maximum"] == pytest.approx(10.0)
    assert summary["mean"] == pytest.approx(5.5)
    assert summary["median"] == pytest.approx(5.5)


def test_the_p95_sits_between_the_mean_and_the_maximum():
    summary = latency_summary_ms(DURATIONS_NS)

    assert summary["mean"] < summary["p95"] <= summary["maximum"]
    # Linear interpolation across ten samples: 1 + 0.95 * 9.
    assert summary["p95"] == pytest.approx(9.55)


def test_a_single_measurement_summarises_to_itself():
    summary = latency_summary_ms([3 * NANOSECONDS_PER_MILLISECOND])

    assert summary == {
        "minimum": pytest.approx(3.0),
        "mean": pytest.approx(3.0),
        "median": pytest.approx(3.0),
        "p95": pytest.approx(3.0),
        "maximum": pytest.approx(3.0),
    }


def test_summarising_no_measurements_is_rejected():
    with pytest.raises(ValueError, match="without any timed inferences"):
        latency_summary_ms([])


def test_total_duration_is_reported_in_seconds():
    # One to ten milliseconds sums to 55 ms.
    assert total_seconds(DURATIONS_NS) == pytest.approx(0.055)


def test_throughput_is_sequences_over_timed_seconds():
    assert throughput_sequences_per_second(500, 2.0) == pytest.approx(250.0)


def test_throughput_requires_a_positive_duration():
    with pytest.raises(ValueError, match="must be positive"):
        throughput_sequences_per_second(10, 0.0)


def test_throughput_rejects_a_negative_sequence_count():
    with pytest.raises(ValueError, match="must not be negative"):
        throughput_sequences_per_second(-1, 1.0)


def test_warmup_is_clamped_to_the_sequences_available():
    assert effective_warmup_calls(100, 500) == 100
    assert effective_warmup_calls(100, 40) == 40
    assert effective_warmup_calls(0, 40) == 0


def test_a_negative_warmup_count_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        effective_warmup_calls(-1, 10)


# ---------------------------------------------------------------------
#                               Timing
# ---------------------------------------------------------------------


def test_timing_measures_one_call_per_sequence(monkeypatch):
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", TickingTimer())
    classifier = CountingClassifier()
    sequences = list(range(5))

    durations_ns = time_inference(
        classifier=classifier,
        sequences=sequences,
        warmup_calls=2,
    )

    # One duration per sequence, and the warm-up calls are not among them.
    assert len(durations_ns) == len(sequences)
    assert classifier.calls == len(sequences) + 2

    # Each timed call spans exactly one clock step.
    assert durations_ns == [NANOSECONDS_PER_MILLISECOND] * len(sequences)


def test_timing_runs_no_warmup_when_none_is_requested(monkeypatch):
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", TickingTimer())
    classifier = CountingClassifier()

    time_inference(
        classifier=classifier,
        sequences=list(range(3)),
        warmup_calls=0,
    )

    assert classifier.calls == 3


def test_timing_without_sequences_is_rejected():
    with pytest.raises(ValueError, match="At least one sequence"):
        time_inference(classifier=CountingClassifier(), sequences=[])


def test_timing_rejects_a_negative_warmup_count():
    with pytest.raises(ValueError, match="must not be negative"):
        time_inference(
            classifier=CountingClassifier(),
            sequences=list(range(3)),
            warmup_calls=-1,
        )
