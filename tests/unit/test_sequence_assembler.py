import numpy as np
import pytest

from ecg_arrhythmia.preprocessing.beat_extraction import (
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    SAMPLING_RATE,
    SEQUENCE_LENGTH,
    WINDOW_SIZE,
)
from ecg_arrhythmia.streaming.sample_buffer import IndexedSampleBuffer
from ecg_arrhythmia.streaming.sequence_assembler import SequenceAssembler

# One beat per second at the dataset sampling rate.
BEAT_SPACING = int(SAMPLING_RATE)
FIRST_PEAK = 500
RECORD_LENGTH = 8000


def _peaks(count: int) -> list[int]:
    return [FIRST_PEAK + index * BEAT_SPACING for index in range(count)]


def _buffer(stop_index: int = RECORD_LENGTH) -> IndexedSampleBuffer:
    buffer = IndexedSampleBuffer()
    buffer.append(np.arange(stop_index, dtype=np.float64), 0)
    return buffer


def test_the_first_detection_only_seeds_the_rr_chain():
    assembler = SequenceAssembler()
    assembler.add_peaks(_peaks(SEQUENCE_LENGTH))

    # Five peaks produce only four beats, one short of a sequence.
    assert assembler.drain(_buffer()) == []


def test_a_full_sequence_has_model_ready_shapes_and_dtypes():
    assembler = SequenceAssembler()
    assembler.add_peaks(_peaks(SEQUENCE_LENGTH + 1))

    [sequence] = assembler.drain(_buffer())

    assert sequence.ecg.shape == (SEQUENCE_LENGTH, 1, WINDOW_SIZE)
    assert sequence.rr.shape == (SEQUENCE_LENGTH, 2)
    assert sequence.ecg.dtype == np.float64
    assert sequence.rr.dtype == np.float64
    assert sequence.target_peak_index == _peaks(SEQUENCE_LENGTH + 1)[-1]
    assert sequence.peak_indices == tuple(_peaks(SEQUENCE_LENGTH + 1)[1:])


def test_windows_hold_exactly_the_samples_around_each_peak():
    assembler = SequenceAssembler()
    assembler.add_peaks(_peaks(SEQUENCE_LENGTH + 1))

    [sequence] = assembler.drain(_buffer())

    for peak, window in zip(sequence.peak_indices, sequence.ecg, strict=True):
        expected = np.arange(peak - SAMPLES_BEFORE, peak + SAMPLES_AFTER)
        np.testing.assert_array_equal(window[0], expected)


def test_rr_features_use_the_shared_helpers():
    assembler = SequenceAssembler()
    assembler.add_peaks(_peaks(SEQUENCE_LENGTH + 1))

    [sequence] = assembler.drain(_buffer())

    # Every interval is exactly one second, so the local rhythm matches it.
    np.testing.assert_allclose(sequence.rr, np.ones((SEQUENCE_LENGTH, 2)))


def test_extraction_waits_for_the_post_peak_samples():
    peaks = _peaks(SEQUENCE_LENGTH + 1)
    assembler = SequenceAssembler()
    assembler.add_peaks(peaks)

    # One sample short of the final beat window.
    partial = _buffer(peaks[-1] + SAMPLES_AFTER - 1)
    assert assembler.drain(partial) == []
    assert assembler.history_start == peaks[-1] - SAMPLES_BEFORE

    assert len(assembler.drain(_buffer())) == 1
    assert assembler.history_start is None


def test_a_peak_without_enough_pre_peak_history_is_skipped():
    assembler = SequenceAssembler()

    # The second peak cannot produce a complete window, but it still
    # advances the RR chain. Five later peaks therefore complete a
    # sequence, which they could not do if the skipped peak had been
    # treated as the RR seed.
    assembler.add_peaks([10, SAMPLES_BEFORE - 1, *_peaks(SEQUENCE_LENGTH)])
    sequences = assembler.drain(_buffer())

    assert len(sequences) == 1
    assert sequences[0].peak_indices == tuple(_peaks(SEQUENCE_LENGTH))


def test_sequences_slide_by_one_beat():
    peaks = _peaks(SEQUENCE_LENGTH + 2)
    assembler = SequenceAssembler()
    assembler.add_peaks(peaks)

    first, second = assembler.drain(_buffer())

    assert first.peak_indices == tuple(peaks[1:-1])
    assert second.peak_indices == tuple(peaks[2:])


def test_peaks_arriving_separately_produce_zero_then_one_sequence():
    peaks = _peaks(SEQUENCE_LENGTH + 1)
    assembler = SequenceAssembler()
    buffer = _buffer()

    for peak in peaks[:-1]:
        assembler.add_peaks([peak])
        assert assembler.drain(buffer) == []

    assembler.add_peaks([peaks[-1]])
    assert len(assembler.drain(buffer)) == 1


def test_end_of_record_drops_a_beat_the_signal_cannot_complete():
    peaks = _peaks(SEQUENCE_LENGTH + 1)
    assembler = SequenceAssembler()
    assembler.add_peaks(peaks)

    truncated = _buffer(peaks[-1] + SAMPLES_AFTER - 1)

    assert assembler.drain(truncated, end_of_record=True) == []
    assert assembler.history_start is None


def test_reset_clears_beat_and_rr_state():
    assembler = SequenceAssembler()
    assembler.add_peaks(_peaks(SEQUENCE_LENGTH + 1))
    assembler.drain(_buffer())

    assembler.reset()

    # Without the retained beats a full run of peaks is needed again.
    assembler.add_peaks(_peaks(SEQUENCE_LENGTH))
    assert assembler.drain(_buffer()) == []


def test_sequence_length_must_be_positive():
    with pytest.raises(ValueError, match="at least one beat"):
        SequenceAssembler(sequence_length=0)
