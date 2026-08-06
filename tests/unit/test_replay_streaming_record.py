import numpy as np
import pytest

from ecg_arrhythmia.evaluation.replay_streaming_record import (
    ReplaySummary,
    replay_record,
)
from ecg_arrhythmia.streaming.replay_source import ReplayMode, ReplaySource
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.streaming.streaming_engine import (
    StreamContinuityError,
    StreamingEngine,
)

SAMPLING_RATE = 360.0


def _source(num_samples=20, chunk_size=4):
    return ReplaySource(
        signal=np.arange(num_samples, dtype=np.float64),
        sampling_rate=SAMPLING_RATE,
        chunk_size=chunk_size,
        mode=ReplayMode.ACCELERATED,
        record_name="synthetic",
        lead_name="MLII",
    )


class CountingEngine(StreamingEngine):
    """Engine that records how many times it was flushed."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_calls = 0

    def flush(self) -> list:
        self.flush_calls += 1

        return super().flush()


class FailingEngine(CountingEngine):
    """Engine that rejects the second chunk of any record."""

    def process_chunk(self, chunk: SampleChunk) -> list:
        if self.state.num_chunks_accepted >= 1:
            raise StreamContinuityError("Simulated continuity failure.")

        return super().process_chunk(chunk)


class TickingClock:
    """Deterministic clock that advances one unit per reading."""

    def __init__(self) -> None:
        self.ticks = 0.0

    def __call__(self) -> float:
        reading = self.ticks
        self.ticks += 1.0

        return reading


class SlowFlushEngine(CountingEngine):
    """Engine whose flush consumes one tick of the replay clock."""

    def __init__(self, clock: TickingClock) -> None:
        super().__init__()
        self._clock = clock

    def flush(self) -> list:
        self._clock()

        return super().flush()


def _patch_replay_clock(monkeypatch, clock: TickingClock) -> None:
    monkeypatch.setattr(
        "ecg_arrhythmia.evaluation.replay_streaming_record.perf_counter",
        clock,
    )


def test_replay_summary_reports_a_complete_stream():
    source = _source(num_samples=20, chunk_size=4)

    summary = replay_record(source=source, engine=StreamingEngine())

    assert summary.record_name == "synthetic"
    assert summary.lead_name == "MLII"
    assert summary.replay_mode == "accelerated"
    assert summary.sampling_rate == SAMPLING_RATE
    assert summary.chunk_size == 4

    assert summary.total_input_samples == 20
    assert summary.total_emitted_chunks == 5
    assert summary.total_samples_accepted == 20
    assert summary.first_sample_index == 0
    assert summary.final_sample_index == 19
    assert summary.continuity_validated is True
    assert summary.elapsed_seconds >= 0.0


def test_replay_summary_includes_a_final_partial_chunk():
    source = _source(num_samples=10, chunk_size=4)

    summary = replay_record(source=source, engine=StreamingEngine())

    assert summary.total_emitted_chunks == 3
    assert summary.total_samples_accepted == 10
    assert summary.final_sample_index == 9


def test_max_samples_stops_the_replay_early():
    source = _source(num_samples=100, chunk_size=4)

    summary = replay_record(
        source=source,
        engine=StreamingEngine(),
        max_samples=12,
    )

    assert summary.total_samples_accepted == 12
    assert summary.total_emitted_chunks == 3
    assert summary.total_input_samples == 100
    assert summary.continuity_validated is True


def test_broken_stream_is_reported_rather_than_hidden():
    source = _source(num_samples=20, chunk_size=4)

    summary = replay_record(source=source, engine=FailingEngine())

    # The failure is visible in the summary, and only the chunks accepted
    # before the failure are counted.
    assert summary.continuity_validated is False
    assert summary.total_emitted_chunks == 1
    assert summary.total_samples_accepted == 4


# ---------------------------------------------------------------------
#                        End-Of-Record Flushing
# ---------------------------------------------------------------------


def test_a_complete_replay_flushes_the_engine_once():
    engine = CountingEngine()

    summary = replay_record(source=_source(num_samples=20, chunk_size=4), engine=engine)

    assert engine.flush_calls == 1
    assert summary.continuity_validated is True
    assert summary.total_samples_accepted == summary.total_input_samples


def test_an_early_limited_replay_is_not_flushed():
    engine = CountingEngine()

    summary = replay_record(
        source=_source(num_samples=100, chunk_size=4),
        engine=engine,
        max_samples=12,
    )

    # The replay stopped short of the end, so it is not a record boundary.
    assert engine.flush_calls == 0
    assert summary.total_samples_accepted == 12


def test_a_limit_equal_to_the_record_length_flushes():
    engine = CountingEngine()

    # The chunk that satisfies the limit is also the chunk that finishes
    # the source, so the record genuinely ended.
    summary = replay_record(
        source=_source(num_samples=20, chunk_size=4),
        engine=engine,
        max_samples=20,
    )

    assert engine.flush_calls == 1
    assert summary.total_samples_accepted == summary.total_input_samples


def test_a_limit_beyond_the_record_length_flushes():
    engine = CountingEngine()

    # The limit is never reached, so the source is exhausted naturally.
    summary = replay_record(
        source=_source(num_samples=20, chunk_size=4),
        engine=engine,
        max_samples=40,
    )

    assert engine.flush_calls == 1
    assert summary.total_samples_accepted == 20


def test_a_partial_final_chunk_that_completes_the_source_flushes():
    engine = CountingEngine()

    # Ten samples in chunks of four ends on a two-sample chunk, which
    # both satisfies the limit and consumes the source.
    summary = replay_record(
        source=_source(num_samples=10, chunk_size=4),
        engine=engine,
        max_samples=10,
    )

    assert engine.flush_calls == 1
    assert summary.total_emitted_chunks == 3
    assert summary.total_samples_accepted == 10


def test_a_broken_stream_is_not_flushed():
    engine = FailingEngine()

    summary = replay_record(source=_source(num_samples=20, chunk_size=4), engine=engine)

    assert engine.flush_calls == 0
    assert summary.continuity_validated is False


def test_elapsed_time_includes_the_flush(monkeypatch):
    clock = TickingClock()
    _patch_replay_clock(monkeypatch, clock)
    engine = SlowFlushEngine(clock)

    summary = replay_record(source=_source(num_samples=8, chunk_size=4), engine=engine)

    # The start reads 0, the flush consumes 1, and the final reading is 2.
    assert engine.flush_calls == 1
    assert summary.elapsed_seconds == pytest.approx(2.0)


def test_elapsed_time_excludes_a_flush_that_never_happens(monkeypatch):
    clock = TickingClock()
    _patch_replay_clock(monkeypatch, clock)
    engine = SlowFlushEngine(clock)

    summary = replay_record(
        source=_source(num_samples=100, chunk_size=4),
        engine=engine,
        max_samples=8,
    )

    assert engine.flush_calls == 0
    assert summary.elapsed_seconds == pytest.approx(1.0)


# ---------------------------------------------------------------------
#                           Summary Ratios
# ---------------------------------------------------------------------


def _summary(
    elapsed_seconds: float,
    total_samples_accepted: int = 720,
) -> ReplaySummary:
    """Build a summary directly so the timing maths is deterministic."""

    return ReplaySummary(
        record_name="synthetic",
        lead_name="MLII",
        replay_mode="accelerated",
        sampling_rate=SAMPLING_RATE,
        chunk_size=360,
        total_input_samples=720,
        total_emitted_chunks=2,
        total_samples_accepted=total_samples_accepted,
        first_sample_index=0,
        final_sample_index=719,
        continuity_validated=True,
        elapsed_seconds=elapsed_seconds,
    )


def test_processed_signal_seconds_uses_accepted_samples():
    # 720 samples at 360 Hz is two seconds of ECG.
    assert _summary(0.5).processed_signal_seconds == pytest.approx(2.0)

    # An early stop is not credited with the whole record.
    assert _summary(
        0.5, total_samples_accepted=360
    ).processed_signal_seconds == pytest.approx(1.0)


def test_real_time_factor_is_wall_time_over_signal_duration():
    # Two seconds of ECG processed in 0.5 s is comfortably faster than
    # real time, so the RTF is below one.
    assert _summary(0.5).real_time_factor == pytest.approx(0.25)

    # Exactly real time.
    assert _summary(2.0).real_time_factor == pytest.approx(1.0)

    # Slower than real time: the system cannot keep up.
    assert _summary(4.0).real_time_factor == pytest.approx(2.0)


def test_speedup_factor_is_the_reciprocal_of_the_real_time_factor():
    assert _summary(0.5).speedup_factor == pytest.approx(4.0)
    assert _summary(2.0).speedup_factor == pytest.approx(1.0)
    assert _summary(4.0).speedup_factor == pytest.approx(0.5)


def test_zero_elapsed_time_is_handled_safely():
    summary = _summary(0.0)

    # No division by zero: the RTF divides by the signal duration, and the
    # speed-up is undefined because no measurable time elapsed.
    assert summary.real_time_factor == pytest.approx(0.0)
    assert summary.speedup_factor is None


def test_ratios_are_undefined_when_no_ecg_was_processed():
    summary = _summary(0.5, total_samples_accepted=0)

    assert summary.processed_signal_seconds == pytest.approx(0.0)
    assert summary.real_time_factor is None
    assert summary.speedup_factor == pytest.approx(0.0)
