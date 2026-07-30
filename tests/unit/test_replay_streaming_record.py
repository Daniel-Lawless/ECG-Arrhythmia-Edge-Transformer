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


class FailingEngine(StreamingEngine):
    """Engine that rejects the second chunk of any record."""

    def process_chunk(self, chunk: SampleChunk) -> None:
        if self.state.num_chunks_accepted >= 1:
            raise StreamContinuityError("Simulated continuity failure.")

        super().process_chunk(chunk)


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
