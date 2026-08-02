import numpy as np
import pytest

from ecg_arrhythmia.streaming.sample_buffer import IndexedSampleBuffer


def _buffer_with(*blocks: list[float], start_index: int = 0) -> IndexedSampleBuffer:
    buffer = IndexedSampleBuffer(start_index)
    next_index = start_index

    for block in blocks:
        buffer.append(np.asarray(block, dtype=np.float64), next_index)
        next_index += len(block)

    return buffer


def test_absolute_indexing_survives_multiple_appends():
    buffer = _buffer_with([0.0, 1.0], [2.0, 3.0], start_index=100)

    assert buffer.start_index == 100
    assert buffer.stop_index == 104
    assert buffer.num_retained == 4
    np.testing.assert_array_equal(buffer.get(102, 104), [2.0, 3.0])


def test_extraction_spans_a_chunk_boundary():
    buffer = _buffer_with([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])

    np.testing.assert_array_equal(buffer.get(2, 5), [2.0, 3.0, 4.0])


def test_append_rejects_a_non_contiguous_block():
    buffer = _buffer_with([0.0, 1.0])

    with pytest.raises(ValueError, match="must continue the buffer at 2"):
        buffer.append(np.asarray([9.0]), 5)


def test_get_rejects_a_range_outside_retained_history():
    buffer = _buffer_with([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="outside the retained history"):
        buffer.get(1, 9)


def test_pruning_keeps_absolute_positions_intact():
    buffer = _buffer_with([0.0, 1.0, 2.0, 3.0, 4.0])

    buffer.prune_before(3)

    assert buffer.start_index == 3
    assert buffer.num_retained == 2
    np.testing.assert_array_equal(buffer.get(3, 5), [3.0, 4.0])

    with pytest.raises(ValueError, match="outside the retained history"):
        buffer.get(2, 4)


def test_pruning_never_discards_unseen_samples():
    buffer = _buffer_with([0.0, 1.0])

    # Far past the newest sample, and far behind the oldest.
    buffer.prune_before(500)
    assert buffer.stop_index == 2
    assert buffer.num_retained == 0

    buffer.append(np.asarray([2.0]), 2)
    buffer.prune_before(0)
    assert buffer.start_index == 2


def test_returned_windows_are_read_only_and_detached():
    buffer = _buffer_with([0.0, 1.0, 2.0, 3.0])
    window = buffer.get(0, 2)

    assert not window.flags.writeable

    with pytest.raises(ValueError):
        window[0] = 99.0

    # A later prune must not disturb an already returned window.
    buffer.prune_before(3)
    np.testing.assert_array_equal(window, [0.0, 1.0])


def test_history_stays_correct_across_lazy_compaction():
    buffer = IndexedSampleBuffer()
    block_size = 100
    next_index = 0

    # Append far past the initial capacity while pruning behind, so the
    # storage is compacted and grown several times along the way.
    for _ in range(40):
        buffer.append(
            np.arange(next_index, next_index + block_size, dtype=np.float64),
            next_index,
        )
        next_index += block_size
        buffer.prune_before(buffer.stop_index - 250)

    assert buffer.num_retained == 250
    np.testing.assert_array_equal(
        buffer.get(buffer.start_index, buffer.stop_index),
        np.arange(buffer.start_index, buffer.stop_index, dtype=np.float64),
    )


def test_reset_rebases_the_buffer():
    buffer = _buffer_with([0.0, 1.0, 2.0])

    buffer.reset(50)

    assert buffer.start_index == 50
    assert buffer.stop_index == 50
    assert buffer.num_retained == 0
