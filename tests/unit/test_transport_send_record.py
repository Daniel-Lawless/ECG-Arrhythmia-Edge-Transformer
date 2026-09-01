import numpy as np
import pytest

from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.transport.protocol import decode_message, encode_runtime_status
from ecg_arrhythmia.transport.send_record import (
    IntervalTimingAccumulator,
    ModelTimingAccumulator,
    TimedClassifier,
    nominal_chunk_period_ms,
    stream_record_to_sender,
)


def _chunk(start_index: int) -> SampleChunk:
    return SampleChunk(
        samples=np.full(36, 0.125, dtype=np.float64),
        start_index=start_index,
        sampling_rate=360.0,
    )


def _event(target: int) -> PredictionEvent:
    return PredictionEvent(
        target_peak_index=target,
        peak_indices=(target - 100, target, target + 100),
        logits=np.array([3.5, -1.25, 0.5, -2.0], dtype=np.float32),
        predicted_class_index=0,
        predicted_label="N",
    )


class FakeSender:
    """Records every send in order, as (kind, identifier, record_name)."""

    def __init__(self) -> None:
        self.sent = []

    def send_sample_chunk(self, chunk, record_name=None) -> None:
        self.sent.append(("sample_chunk", chunk.start_index, record_name))

    def send_prediction(self, event, record_name=None) -> None:
        self.sent.append(("prediction", event.target_peak_index, record_name))

    def send_runtime_status(self, status) -> None:
        self.sent.append(("runtime_status", status, None))

    def statuses(self) -> list:
        return [entry[1] for entry in self.sent if entry[0] == "runtime_status"]


class FakeSource:
    def __init__(self, chunks, record_name="114") -> None:
        self.record_name = record_name
        self._chunks = list(chunks)

    def iter_chunks(self):
        yield from self._chunks


class FakePredictor:
    """Scripted events per chunk, mirroring StreamingPredictor's API."""

    def __init__(self, events_per_chunk, flush_events) -> None:
        self._events_per_chunk = list(events_per_chunk)
        self._flush_events = list(flush_events)
        self.started_with = None

    def start_record(self, record_name=None, start_index=0) -> None:
        self.started_with = record_name

    def process_chunk(self, chunk):
        return self._events_per_chunk.pop(0)

    def flush(self):
        return self._flush_events


def test_each_chunk_is_sent_before_the_predictions_it_produced():
    sender = FakeSender()
    source = FakeSource([_chunk(0), _chunk(36), _chunk(72)])
    predictor = FakePredictor(
        events_per_chunk=[[], [_event(40)], []],
        flush_events=[_event(100)],
    )

    summary = stream_record_to_sender(sender, source, predictor)

    # Documented ordering contract: each sample_chunk precedes the
    # predictions its processing produced, and flush predictions are
    # transmitted after every chunk.
    assert sender.sent == [
        ("sample_chunk", 0, "114"),
        ("sample_chunk", 36, "114"),
        ("prediction", 40, "114"),
        ("sample_chunk", 72, "114"),
        ("prediction", 100, "114"),
    ]
    assert summary == {
        "chunks_sent": 3,
        "predictions_sent": 1,
        "flush_predictions_sent": 1,
        "runtime_statuses_sent": 0,
        "stopped_early": False,
    }


def test_the_record_is_started_with_the_source_record_name():
    sender = FakeSender()
    source = FakeSource([_chunk(0)], record_name="233")
    predictor = FakePredictor(events_per_chunk=[[]], flush_events=[])

    stream_record_to_sender(sender, source, predictor)

    assert predictor.started_with == "233"
    assert sender.sent == [("sample_chunk", 0, "233")]


def test_a_chunk_completing_several_sequences_sends_them_in_order():
    sender = FakeSender()
    source = FakeSource([_chunk(0)])
    predictor = FakePredictor(
        events_per_chunk=[[_event(10), _event(20)]],
        flush_events=[],
    )

    stream_record_to_sender(sender, source, predictor)

    assert sender.sent == [
        ("sample_chunk", 0, "114"),
        ("prediction", 10, "114"),
        ("prediction", 20, "114"),
    ]


# ---------------------------------------------------------------------
#                    Runtime Telemetry (Section 6.2.5)
# ---------------------------------------------------------------------


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeTimer:
    """
    Scripted perf_counter_ns: each chunk consumes one (start, stop)
    pair, so the n-th chunk's process_chunk() appears to take the n-th
    scripted duration.
    """

    def __init__(self, durations_ms) -> None:
        self._values = []
        now_ns = 0

        for duration_ms in durations_ms:
            self._values.append(now_ns)
            now_ns += int(duration_ms * 1_000_000)
            self._values.append(now_ns)

    def __call__(self) -> int:
        return self._values.pop(0)


class TickingSource:
    """Advances the fake monotonic clock per chunk, like paced replay."""

    def __init__(
        self,
        chunks,
        clock,
        tick_seconds,
        record_name="114",
        chunk_size=36,
        sampling_rate=360.0,
    ) -> None:
        self.record_name = record_name
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self._chunks = list(chunks)
        self._clock = clock
        self._tick = tick_seconds

    def iter_chunks(self):
        for chunk in self._chunks:
            self._clock.value += self._tick
            yield chunk


class FakeTelemetry:
    def __init__(self) -> None:
        self.samples_taken = 0

    def sample(self) -> dict:
        self.samples_taken += 1

        return {
            "temperature_c": 48.7,
            "process_cpu_percent": 3.5,
            "process_rss_mib": 253.0,
            "available_ram_mib": 610.0,
            "cpu_frequency_mhz": 2400.0,
            "cpu_governor": "performance",
            "under_voltage_active": False,
            "frequency_capped_active": False,
            "throttling_active": False,
            "soft_temp_limit_active": False,
            "runtime_condition_occurred": False,
        }


def _quiet_predictor(num_chunks: int) -> FakePredictor:
    return FakePredictor(events_per_chunk=[[]] * num_chunks, flush_events=[])


def test_the_nominal_chunk_period_derives_from_the_source():
    source = TickingSource([], FakeClock(), 0.1)

    # 36 samples at 360 Hz: 100 ms, derived rather than hard-coded.
    assert nominal_chunk_period_ms(source) == pytest.approx(100.0)


def test_interval_timing_accumulator_tracks_the_window_maximum():
    accumulator = IntervalTimingAccumulator()

    for duration_ms in (0.02, 0.03, 4.5, 0.01):
        accumulator.record(duration_ms)

    assert accumulator.window_max_ms == pytest.approx(4.5)
    assert accumulator.latest_ms == pytest.approx(0.01)

    accumulator.reset()

    assert accumulator.window_max_ms is None
    assert accumulator.latest_ms is None


def test_runtime_status_cadence_follows_the_injected_clock():
    clock = FakeClock(0.0)
    chunks = [_chunk(start) for start in range(0, 216, 36)]
    source = TickingSource(chunks, clock, tick_seconds=0.4)
    sender = FakeSender()

    summary = stream_record_to_sender(
        sender,
        source,
        _quiet_predictor(len(chunks)),
        telemetry=FakeTelemetry(),
        status_interval_seconds=1.0,
        clock=clock,
        timer=FakeTimer([1.0] * len(chunks)),
    )

    kinds = [entry[0] for entry in sender.sent]

    # Six chunks at 0.4 s each with a 1.0 s interval: statuses become
    # due after the third and sixth chunks, after those chunks'
    # messages, preserving the documented ordering.
    assert kinds == [
        "sample_chunk",
        "sample_chunk",
        "sample_chunk",
        "runtime_status",
        "sample_chunk",
        "sample_chunk",
        "sample_chunk",
        "runtime_status",
    ]
    assert summary["runtime_statuses_sent"] == 2


def test_window_maximum_and_headroom_derive_from_processing_times():
    clock = FakeClock(0.0)
    chunks = [_chunk(start) for start in range(0, 216, 36)]
    source = TickingSource(chunks, clock, tick_seconds=0.4)
    sender = FakeSender()
    telemetry = FakeTelemetry()

    stream_record_to_sender(
        sender,
        source,
        _quiet_predictor(len(chunks)),
        telemetry=telemetry,
        status_interval_seconds=1.0,
        clock=clock,
        timer=FakeTimer([0.02, 0.03, 4.5, 0.01, 2.0, 0.5]),
    )

    first, second = sender.statuses()

    # First interval: chunks 1-3, worst case 4.5 ms of the 100 ms
    # nominal period.
    assert first["window_max_chunk_processing_ms"] == pytest.approx(4.5)
    assert first["window_min_processing_headroom_ms"] == pytest.approx(95.5)
    # The accumulator reset after emission: the second interval only
    # saw chunks 4-6.
    assert second["window_max_chunk_processing_ms"] == pytest.approx(2.0)
    assert second["window_min_processing_headroom_ms"] == pytest.approx(98.0)

    # Stream context and hardware fields are merged into the status.
    assert first["record_name"] == "114"
    assert first["latest_sample_index"] == chunks[2].last_index
    assert first["temperature_c"] == 48.7
    assert first["cpu_governor"] == "performance"
    assert telemetry.samples_taken == 2


def test_negative_processing_headroom_is_not_clamped():
    clock = FakeClock(0.0)
    source = TickingSource([_chunk(0)], clock, tick_seconds=1.5)
    sender = FakeSender()

    stream_record_to_sender(
        sender,
        source,
        _quiet_predictor(1),
        telemetry=FakeTelemetry(),
        status_interval_seconds=1.0,
        clock=clock,
        timer=FakeTimer([150.0]),
    )

    (status,) = sender.statuses()

    assert status["window_max_chunk_processing_ms"] == pytest.approx(150.0)
    assert status["window_min_processing_headroom_ms"] == pytest.approx(-50.0)


# ---------------------------------------------------------------------
#            Cooperative Stop (dashboard record control)
# ---------------------------------------------------------------------


def test_the_stream_runs_to_completion_when_no_stop_is_supplied():
    # The default path must stay exactly as it was before the stop
    # parameter existed.
    sender = FakeSender()
    chunks = [_chunk(start) for start in range(0, 108, 36)]
    source = FakeSource(chunks)

    summary = stream_record_to_sender(
        sender,
        source,
        FakePredictor(events_per_chunk=[[]] * 3, flush_events=[_event(50)]),
    )

    assert summary["chunks_sent"] == 3
    assert summary["stopped_early"] is False


def test_a_stop_request_ends_the_replay_at_a_chunk_boundary():
    sender = FakeSender()
    chunks = [_chunk(start) for start in range(0, 180, 36)]
    source = FakeSource(chunks)
    calls = {"count": 0}

    def should_stop() -> bool:
        # Stop before the third chunk is sent.
        calls["count"] += 1

        return calls["count"] > 2

    summary = stream_record_to_sender(
        sender,
        source,
        FakePredictor(events_per_chunk=[[]] * 5, flush_events=[_event(99)]),
        should_stop=should_stop,
    )

    assert summary["chunks_sent"] == 2
    assert summary["stopped_early"] is True
    # Flush still ran, so the stream ends on a coherent message
    # sequence rather than mid-record.
    assert summary["flush_predictions_sent"] == 1
    assert sender.sent[-1] == ("prediction", 99, "114")


def test_an_immediate_stop_sends_nothing_but_still_flushes():
    sender = FakeSender()
    source = FakeSource([_chunk(0), _chunk(36)])

    summary = stream_record_to_sender(
        sender,
        source,
        FakePredictor(events_per_chunk=[[], []], flush_events=[]),
        should_stop=lambda: True,
    )

    assert summary["chunks_sent"] == 0
    assert summary["stopped_early"] is True
    assert sender.sent == []


# ---------------------------------------------------------------------
#              Model-Stage Timing (Section 6.3 Final Addition)
# ---------------------------------------------------------------------

NS_PER_MS = 1_000_000


def test_an_empty_model_accumulator_reports_null_never_zero():
    accumulator = ModelTimingAccumulator()

    assert accumulator.calls == 0
    assert accumulator.sequences == 0
    assert accumulator.mean_latency_ms is None
    assert accumulator.throughput_sequences_per_second is None


def test_one_model_observation_defines_both_metrics():
    accumulator = ModelTimingAccumulator()

    accumulator.record(2 * NS_PER_MS)

    # One sequence in 2 ms: mean 2 ms, 500 sequences per timed second.
    assert accumulator.mean_latency_ms == pytest.approx(2.0)
    assert accumulator.throughput_sequences_per_second == pytest.approx(500.0)


def test_multiple_observations_aggregate_by_total_timed_work():
    accumulator = ModelTimingAccumulator()

    for elapsed_ms in (1.0, 2.0, 3.0):
        accumulator.record(int(elapsed_ms * NS_PER_MS))

    # 3 sequences in 6 ms total: mean 2 ms, 500 seq/s - derived from
    # totals, not from averaging per-call rates.
    assert accumulator.calls == 3
    assert accumulator.sequences == 3
    assert accumulator.mean_latency_ms == pytest.approx(2.0)
    assert accumulator.throughput_sequences_per_second == pytest.approx(500.0)


def test_throughput_counts_sequences_rather_than_assuming_one_per_call():
    accumulator = ModelTimingAccumulator()

    accumulator.record(4 * NS_PER_MS, sequences=3)

    # 3 sequences in 4 ms: capacity reflects the measured sequence
    # count, so the definition stays correct if batching ever changes.
    assert accumulator.calls == 1
    assert accumulator.sequences == 3
    assert accumulator.throughput_sequences_per_second == pytest.approx(750.0)


def test_model_accumulator_reset_returns_it_to_the_null_state():
    accumulator = ModelTimingAccumulator()
    accumulator.record(NS_PER_MS)

    accumulator.reset()

    assert accumulator.calls == 0
    assert accumulator.mean_latency_ms is None
    assert accumulator.throughput_sequences_per_second is None


class SeamRecordingClassifier:
    """Logs its predict() calls and returns a unique sentinel object."""

    def __init__(self, log) -> None:
        self._log = log
        self.result = object()

    def predict(self, sequence):
        self._log.append(("predict", sequence))

        return self.result


class LoggingAccumulator(ModelTimingAccumulator):
    def __init__(self, log) -> None:
        super().__init__()
        self._log = log

    def record(self, elapsed_ns, sequences=1) -> None:
        self._log.append(("record", elapsed_ns))
        super().record(elapsed_ns, sequences)


def test_timed_classifier_times_exactly_the_wrapped_predict_call():
    log = []
    readings = iter((0, 5 * NS_PER_MS))

    def timer():
        log.append(("timer", None))

        return next(readings)

    inner = SeamRecordingClassifier(log)
    accumulator = LoggingAccumulator(log)
    sequence = object()

    result = TimedClassifier(inner, accumulator, timer=timer).predict(sequence)

    # The seam: timer, the inner predict alone, timer - and the
    # accumulator bookkeeping strictly AFTER the measured region.
    assert [entry[0] for entry in log] == ["timer", "predict", "timer", "record"]
    assert log[1] == ("predict", sequence)
    # Elapsed = second scripted reading minus the first (5 ms).
    assert log[3][1] == 5 * NS_PER_MS
    # The wrapper returns the inner classifier's result exactly.
    assert result is inner.result


class ModelRecordingPredictor(FakePredictor):
    """
    Scripted model-stage observations per chunk, mimicking what a
    TimedClassifier inside the real predictor would feed the shared
    accumulator during process_chunk().
    """

    def __init__(self, events_per_chunk, accumulator, elapsed_ns_per_chunk) -> None:
        super().__init__(events_per_chunk, flush_events=[])
        self._accumulator = accumulator
        self._elapsed = list(elapsed_ns_per_chunk)

    def process_chunk(self, chunk):
        for elapsed_ns in self._elapsed.pop(0):
            self._accumulator.record(elapsed_ns)

        return super().process_chunk(chunk)


def _model_stream(clock, model_timing, elapsed_ns_per_chunk):
    """Six 0.4 s chunks, 1 s statuses: emissions after chunks 3 and 6."""

    chunks = [_chunk(start) for start in range(0, 216, 36)]
    source = TickingSource(chunks, clock, tick_seconds=0.4)
    sender = FakeSender()
    predictor = ModelRecordingPredictor(
        events_per_chunk=[[]] * len(chunks),
        accumulator=model_timing,
        elapsed_ns_per_chunk=elapsed_ns_per_chunk,
    )

    stream_record_to_sender(
        sender,
        source,
        predictor,
        telemetry=FakeTelemetry(),
        model_timing=model_timing,
        status_interval_seconds=1.0,
        clock=clock,
        timer=FakeTimer([1.0] * len(chunks)),
    )

    return sender.statuses()


def test_an_interval_with_inference_reports_fresh_model_values():
    clock = FakeClock(0.0)
    model_timing = ModelTimingAccumulator()

    # Two inferences during chunk 2 (1.5 ms and 2.5 ms), then quiet.
    first, second = _model_stream(
        clock,
        model_timing,
        [[], [1_500_000, 2_500_000], [], [], [], []],
    )

    # 2 sequences in 4 ms: mean 2 ms, 500 seq/s, fresh at emission.
    assert first["model_inference_mean_ms"] == pytest.approx(2.0)
    assert first["model_throughput_sequences_per_second"] == pytest.approx(500.0)
    assert first["model_measurement_age_seconds"] == pytest.approx(0.0)

    # Quiet second interval: values retained (never flashed back to
    # null), dated by the growing measurement age (1.2 s of chunks).
    assert second["model_inference_mean_ms"] == pytest.approx(2.0)
    assert second["model_throughput_sequences_per_second"] == pytest.approx(500.0)
    assert second["model_measurement_age_seconds"] == pytest.approx(1.2)


def test_model_fields_are_null_before_the_first_ever_inference():
    clock = FakeClock(0.0)
    model_timing = ModelTimingAccumulator()

    # Inference first occurs during chunk 5, in the second interval.
    first, second = _model_stream(
        clock,
        model_timing,
        [[], [], [], [], [2_000_000], []],
    )

    assert first["model_inference_mean_ms"] is None
    assert first["model_throughput_sequences_per_second"] is None
    assert first["model_measurement_age_seconds"] is None

    assert second["model_inference_mean_ms"] == pytest.approx(2.0)
    assert second["model_measurement_age_seconds"] == pytest.approx(0.0)


def test_the_model_accumulator_resets_after_each_captured_interval():
    clock = FakeClock(0.0)
    model_timing = ModelTimingAccumulator()

    # Both intervals contain inference; the second reports ONLY its
    # own interval's work, proving the capture reset the accumulator.
    first, second = _model_stream(
        clock,
        model_timing,
        [[1_000_000], [], [], [], [4_000_000], []],
    )

    assert first["model_inference_mean_ms"] == pytest.approx(1.0)
    assert second["model_inference_mean_ms"] == pytest.approx(4.0)
    assert second["model_measurement_age_seconds"] == pytest.approx(0.0)


def test_statuses_without_model_timing_still_carry_null_model_fields():
    clock = FakeClock(0.0)
    source = TickingSource([_chunk(0)], clock, tick_seconds=1.5)
    sender = FakeSender()

    stream_record_to_sender(
        sender,
        source,
        _quiet_predictor(1),
        telemetry=FakeTelemetry(),
        status_interval_seconds=1.0,
        clock=clock,
        timer=FakeTimer([1.0]),
    )

    (status,) = sender.statuses()

    assert status["model_inference_mean_ms"] is None
    assert status["model_throughput_sequences_per_second"] is None
    assert status["model_measurement_age_seconds"] is None


def test_the_loop_status_dict_matches_the_protocol_encoder_exactly():
    clock = FakeClock(0.0)
    model_timing = ModelTimingAccumulator()

    first, second = _model_stream(
        clock,
        model_timing,
        [[], [1_500_000, 2_500_000], [], [], [], []],
    )

    # Every status dict the loop builds must encode and round-trip
    # through the real protocol without adaptation.
    for status in (first, second):
        message = decode_message(encode_runtime_status(**status))

        assert message["message_type"] == "runtime_status"
        assert message["model_inference_mean_ms"] == pytest.approx(2.0)
