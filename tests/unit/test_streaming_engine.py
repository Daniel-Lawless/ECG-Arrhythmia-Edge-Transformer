import numpy as np
import pytest

from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.streaming.streaming_engine import (
    StreamContinuityError,
    StreamingEngine,
)

SAMPLING_RATE = 360.0


def _chunk(start_index, num_samples=4, sampling_rate=SAMPLING_RATE):
    return SampleChunk(
        samples=np.arange(num_samples, dtype=np.float64),
        start_index=start_index,
        sampling_rate=sampling_rate,
    )


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

    # The new record starts from sample zero and may use another rate,
    # so no state leaks across the record boundary.
    engine.process_chunk(_chunk(0, sampling_rate=250.0))

    state = engine.state
    assert state.record_name == "122"
    assert state.num_chunks_accepted == 1
    assert state.total_samples_accepted == 4
    assert state.sampling_rate == 250.0


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
