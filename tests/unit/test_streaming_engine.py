import numpy as np
import pytest

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SAMPLING_RATE as MODEL_SAMPLING_RATE,
)
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SEQUENCE_LENGTH,
    WINDOW_SIZE,
)
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.streaming.streaming_detector import DetectorTiming
from ecg_arrhythmia.streaming.streaming_engine import (
    StreamContinuityError,
    StreamingEngine,
)

SAMPLING_RATE = 360.0

# Fast timings so a synthetic record exercises the whole pipeline without
# needing thirty seconds of ECG.
FAST_TIMING = DetectorTiming(
    analysis_window_seconds=30.0,
    stride_seconds=1.0,
    warmup_seconds=1.0,
    confirmation_seconds=0.5,
)
SYNTHETIC_LENGTH = 6000
SYNTHETIC_PEAKS = [300, 660, 1020, 1380, 1740, 2100, 2460]


def _chunk(start_index, num_samples=4, sampling_rate=SAMPLING_RATE):
    return SampleChunk(
        samples=np.arange(num_samples, dtype=np.float64),
        start_index=start_index,
        sampling_rate=sampling_rate,
    )


class _MarkerDetector(RPeakDetector):
    """Fake detector that treats every sample equal to 1.0 as an R-peak."""

    @property
    def name(self):
        return "marker"

    def _detect(self, signal, sampling_rate):
        return np.flatnonzero(signal == 1.0).astype(np.int64)


def _synthetic_record():
    signal = np.zeros(SYNTHETIC_LENGTH, dtype=np.float64)
    signal[SYNTHETIC_PEAKS] = 1.0
    return signal


def _synthetic_engine():
    engine = StreamingEngine(detector=_MarkerDetector(), timing=FAST_TIMING)
    engine.start_record("synthetic")
    return engine


def _stream(engine, signal, chunk_size=36):
    sequences = []

    for start_index in range(0, len(signal), chunk_size):
        chunk = SampleChunk(
            samples=signal[start_index : start_index + chunk_size],
            start_index=start_index,
            sampling_rate=SAMPLING_RATE,
        )
        sequences.extend(engine.process_chunk(chunk))

    sequences.extend(engine.flush())

    return sequences


# ---------------------------------------------------------------------
#                         Accepted Streams
# ---------------------------------------------------------------------


def test_engine_accepts_contiguous_chunks():
    engine = StreamingEngine()
    engine.start_record("114")

    for start_index in (0, 4, 8):
        engine.process_chunk(_chunk(start_index))

    state = engine.state
    assert state.record_name == "114"
    assert state.num_chunks_accepted == 3
    assert state.total_samples_accepted == 12
    assert state.next_expected_index == 12
    assert state.first_sample_index == 0
    assert state.last_sample_index == 11
    assert state.sampling_rate == SAMPLING_RATE


def test_engine_accepts_a_final_partial_chunk():
    engine = StreamingEngine()
    engine.start_record("114")

    engine.process_chunk(_chunk(0, num_samples=4))
    engine.process_chunk(_chunk(4, num_samples=2))

    assert engine.state.total_samples_accepted == 6
    assert engine.state.last_sample_index == 5


def test_fresh_engine_has_empty_state():
    state = StreamingEngine().state

    assert state.record_name is None
    assert state.sampling_rate is None
    assert state.num_chunks_accepted == 0
    assert state.total_samples_accepted == 0
    assert state.next_expected_index == 0
    assert state.first_sample_index is None
    assert state.last_sample_index is None


# ---------------------------------------------------------------------
#                        Rejected Streams
# ---------------------------------------------------------------------


def test_engine_rejects_a_gap():
    engine = StreamingEngine()
    engine.process_chunk(_chunk(0))

    # Sample 4 is missing entirely.
    with pytest.raises(StreamContinuityError, match="Gap of 1 samples"):
        engine.process_chunk(_chunk(5))


def test_engine_rejects_an_overlap():
    engine = StreamingEngine()
    engine.process_chunk(_chunk(0))

    # Starts before the stream position and extends past it.
    with pytest.raises(StreamContinuityError, match="Overlapping or out-of-order"):
        engine.process_chunk(_chunk(2))


def test_engine_rejects_a_duplicate_chunk():
    engine = StreamingEngine()
    engine.process_chunk(_chunk(0))

    with pytest.raises(StreamContinuityError, match="Duplicate chunk"):
        engine.process_chunk(_chunk(0))


def test_engine_rejects_an_out_of_order_chunk():
    engine = StreamingEngine()
    engine.process_chunk(_chunk(0))
    engine.process_chunk(_chunk(4))

    # An old chunk arriving late is not the duplicate of the previous one.
    with pytest.raises(StreamContinuityError, match="Overlapping or out-of-order"):
        engine.process_chunk(_chunk(0))


def test_engine_rejects_a_sampling_rate_change():
    engine = StreamingEngine()
    engine.process_chunk(_chunk(0, sampling_rate=360.0))

    with pytest.raises(StreamContinuityError, match="Sampling rate changed"):
        engine.process_chunk(_chunk(4, sampling_rate=250.0))


# ---------------------------------------------------------------------
#                     Model Sampling-Rate Contract
# ---------------------------------------------------------------------


def test_engine_accepts_the_model_sampling_rate():
    engine = StreamingEngine()

    engine.process_chunk(_chunk(0, sampling_rate=MODEL_SAMPLING_RATE))

    assert engine.state.sampling_rate == MODEL_SAMPLING_RATE


def test_engine_rejects_a_first_chunk_at_another_sampling_rate():
    engine = StreamingEngine()

    with pytest.raises(ValueError, match=r"requires 360 Hz input") as error:
        engine.process_chunk(_chunk(0, sampling_rate=250.0))

    # An unusable rate is a model-contract violation, not a broken stream.
    assert not isinstance(error.value, StreamContinuityError)


def test_a_rejected_sampling_rate_leaves_every_stage_untouched():
    engine = _synthetic_engine()

    with pytest.raises(ValueError, match=r"requires 360 Hz input"):
        engine.process_chunk(_chunk(0, num_samples=36, sampling_rate=128.0))

    state = engine.state
    assert state.sampling_rate is None
    assert state.num_chunks_accepted == 0
    assert state.total_samples_accepted == 0
    assert state.next_expected_index == 0
    assert state.first_sample_index is None
    assert engine.last_confirmed_peaks == ()

    # The buffer, detector and assembler are untouched too, so the whole
    # record still streams from sample zero and produces its sequences.
    sequences = _stream(engine, _synthetic_record())

    assert [sequence.target_peak_index for sequence in sequences] == (
        SYNTHETIC_PEAKS[-2:]
    )


def test_rejected_chunk_leaves_state_unchanged():
    engine = StreamingEngine()
    engine.process_chunk(_chunk(0))

    with pytest.raises(StreamContinuityError):
        engine.process_chunk(_chunk(99))

    # The failed chunk must not be counted or advance the stream.
    assert engine.state.num_chunks_accepted == 1
    assert engine.state.total_samples_accepted == 4
    assert engine.state.next_expected_index == 4


# ---------------------------------------------------------------------
#                         Record Boundaries
# ---------------------------------------------------------------------


def test_reset_clears_stream_state():
    engine = StreamingEngine()
    engine.start_record("114")
    engine.process_chunk(_chunk(0))
    engine.process_chunk(_chunk(4))

    engine.reset()

    state = engine.state
    assert state.record_name is None
    assert state.num_chunks_accepted == 0
    assert state.total_samples_accepted == 0
    assert state.next_expected_index == 0
    assert state.first_sample_index is None
    assert state.sampling_rate is None


def test_starting_a_new_record_allows_a_fresh_stream():
    engine = StreamingEngine()
    engine.start_record("114")
    engine.process_chunk(_chunk(0))
    engine.process_chunk(_chunk(4))

    engine.start_record("122")

    # The new record starts again from sample zero, so no counters or
    # stream position leak across the record boundary.
    engine.process_chunk(_chunk(0))

    state = engine.state
    assert state.record_name == "122"
    assert state.num_chunks_accepted == 1
    assert state.total_samples_accepted == 4
    assert state.first_sample_index == 0
    assert state.sampling_rate == SAMPLING_RATE


def test_engine_accepts_a_stream_that_does_not_start_at_zero():
    # A source that deliberately begins mid-record declares its starting
    # position, so callers never touch internal continuity state.
    engine = StreamingEngine()
    engine.start_record(record_name="114", start_index=1000)

    engine.process_chunk(_chunk(1000))
    engine.process_chunk(_chunk(1004))

    state = engine.state
    assert state.record_name == "114"
    assert state.first_sample_index == 1000
    assert state.last_sample_index == 1007
    assert state.total_samples_accepted == 8


def test_start_index_is_enforced_for_the_first_chunk():
    engine = StreamingEngine()
    engine.start_record(record_name="114", start_index=1000)

    # Starting at zero no longer continues this stream.
    with pytest.raises(StreamContinuityError, match="Overlapping or out-of-order"):
        engine.process_chunk(_chunk(0))


def test_start_record_defaults_to_the_beginning_of_the_record():
    engine = StreamingEngine()
    engine.start_record(record_name="114")

    assert engine.state.next_expected_index == 0


def test_start_record_rejects_a_negative_start_index():
    engine = StreamingEngine()

    with pytest.raises(ValueError):
        engine.start_record(record_name="114", start_index=-1)


def test_start_record_rejects_a_non_integer_start_index():
    engine = StreamingEngine()

    with pytest.raises(TypeError):
        engine.start_record(record_name="114", start_index=1.5)


# ---------------------------------------------------------------------
#                        Model-Ready Sequences
# ---------------------------------------------------------------------


def test_no_sequences_are_produced_before_warm_up():
    engine = _synthetic_engine()

    assert engine.process_chunk(_chunk(0, num_samples=36)) == []
    assert engine.last_confirmed_peaks == ()


def test_a_replayed_record_produces_model_ready_sequences():
    sequences = _stream(_synthetic_engine(), _synthetic_record())

    assert [sequence.target_peak_index for sequence in sequences] == (
        SYNTHETIC_PEAKS[-2:]
    )

    for sequence in sequences:
        assert sequence.ecg.shape == (SEQUENCE_LENGTH, 1, WINDOW_SIZE)
        assert sequence.rr.shape == (SEQUENCE_LENGTH, 2)
        assert len(sequence.peak_indices) == SEQUENCE_LENGTH


def test_a_rejected_chunk_does_not_advance_processing_state():
    engine = _synthetic_engine()
    signal = _synthetic_record()

    engine.process_chunk(SampleChunk(signal[0:36], 0, SAMPLING_RATE))

    with pytest.raises(StreamContinuityError):
        engine.process_chunk(SampleChunk(signal[72:108], 72, SAMPLING_RATE))

    # The buffer is still positioned at 36, so the correct chunk still fits.
    engine.process_chunk(SampleChunk(signal[36:72], 36, SAMPLING_RATE))

    assert engine.state.total_samples_accepted == 72


def test_starting_a_new_record_clears_all_section_two_state():
    engine = _synthetic_engine()
    signal = _synthetic_record()

    first = _stream(engine, signal)
    engine.start_record("synthetic-again")
    second = _stream(engine, signal)

    assert [sequence.target_peak_index for sequence in second] == [
        sequence.target_peak_index for sequence in first
    ]
    np.testing.assert_array_equal(second[0].ecg, first[0].ecg)
    np.testing.assert_array_equal(second[0].rr, first[0].rr)


def test_flush_before_any_chunk_returns_nothing():
    assert StreamingEngine().flush() == []
