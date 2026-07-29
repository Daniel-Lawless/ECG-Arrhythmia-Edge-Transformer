import dataclasses

import numpy as np
import pytest

from ecg_arrhythmia.streaming.sample_chunk import SampleChunk

SAMPLING_RATE = 360.0


# Initalise a chunk
def _chunk(samples, start_index=0, sampling_rate=SAMPLING_RATE):
    return SampleChunk(
        samples=np.asarray(samples, dtype=np.float64),
        start_index=start_index,
        sampling_rate=sampling_rate,
    )


def test_chunk_exposes_absolute_positions():
    chunk = _chunk(np.arange(4), start_index=100)

    assert chunk.num_samples == 4
    assert chunk.start_index == 100
    assert chunk.stop_index == 104
    assert chunk.last_index == 103
    assert chunk.duration_seconds == pytest.approx(4 / SAMPLING_RATE)


def test_chunk_samples_are_read_only():
    chunk = _chunk(np.arange(4))

    # Downstream stages must not be able to mutate a chunk in place.
    with pytest.raises(ValueError):
        chunk.samples[0] = 99.0


def test_chunk_is_frozen():
    chunk = _chunk(np.arange(4))

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.start_index = 5


def test_chunk_does_not_copy_the_underlying_buffer():
    signal = np.arange(8, dtype=np.float64)

    chunk = _chunk(signal[2:6], start_index=2)

    # A view is taken rather than a copy, so the chunk shares memory with
    # the source signal instead of duplicating every block.
    assert np.shares_memory(chunk.samples, signal)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((2, 4)),
        np.zeros((4, 1)),
    ],
)
def test_chunk_rejects_non_one_dimensional_samples(samples):
    with pytest.raises(ValueError):
        _chunk(samples)


def test_chunk_rejects_empty_samples():
    with pytest.raises(ValueError):
        _chunk(np.array([]))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_chunk_rejects_non_finite_samples(bad_value):
    with pytest.raises(ValueError):
        _chunk(np.array([0.0, bad_value, 1.0]))


def test_chunk_rejects_negative_start_index():
    with pytest.raises(ValueError):
        _chunk(np.arange(4), start_index=-1)


def test_chunk_rejects_non_integer_start_index():
    with pytest.raises(TypeError):
        _chunk(np.arange(4), start_index=1.5)


@pytest.mark.parametrize("sampling_rate", [0, -360, np.nan, np.inf])
def test_chunk_rejects_invalid_sampling_rate(sampling_rate):
    with pytest.raises(ValueError):
        _chunk(np.arange(4), sampling_rate=sampling_rate)
