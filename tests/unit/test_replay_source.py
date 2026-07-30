import numpy as np
import pytest

from ecg_arrhythmia.streaming.replay_source import ReplayMode, ReplaySource

SAMPLING_RATE = 360.0


class FakeClock:
    """
    Deterministic monotonic clock whose time only advances when the
    injected sleep is called, or when a test advances it explicitly.

    This lets the real-time scheduling logic be verified exactly without
    any test having to wait.
    """

    def __init__(self, start_time: float = 1000.0) -> None:
        self.now = start_time
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        """Simulate time spent by the consumer between chunks."""

        self.now += seconds


def _source(signal, chunk_size=4, mode=ReplayMode.ACCELERATED, clock=None):
    fake_clock = clock or FakeClock()
    return ReplaySource(
        signal=np.asarray(signal, dtype=np.float64),
        sampling_rate=SAMPLING_RATE,
        chunk_size=chunk_size,
        mode=mode,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )


# ---------------------------------------------------------------------
#                        Accelerated Chunking
# ---------------------------------------------------------------------


def test_accelerated_replay_emits_contiguous_chunks():
    source = _source(np.arange(10), chunk_size=4)

    chunks = list(source.iter_chunks())

    assert len(chunks) == 3
    assert source.num_chunks == 3
    assert [chunk.start_index for chunk in chunks] == [0, 4, 8]
    assert [chunk.num_samples for chunk in chunks] == [4, 4, 2]


def test_replay_preserves_every_sample_exactly():
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(37)

    source = _source(signal, chunk_size=5)
    chunks = list(source.iter_chunks())

    # No sample is lost, duplicated, reordered or altered.
    reassembled = np.concatenate([chunk.samples for chunk in chunks])
    np.testing.assert_array_equal(reassembled, signal)

    # Absolute indices tile the record with no gaps or overlaps.
    expected_index = 0
    for chunk in chunks:
        assert chunk.start_index == expected_index
        expected_index = chunk.stop_index
    assert expected_index == signal.size


def test_final_partial_chunk_is_included():
    source = _source(np.arange(10), chunk_size=4)

    final_chunk = list(source.iter_chunks())[-1]

    assert final_chunk.num_samples == 2
    assert final_chunk.start_index == 8
    np.testing.assert_array_equal(final_chunk.samples, np.array([8.0, 9.0]))


def test_signal_shorter_than_one_chunk_emits_one_chunk():
    source = _source(np.arange(3), chunk_size=10)

    chunks = list(source.iter_chunks())

    assert len(chunks) == 1
    assert chunks[0].num_samples == 3
    assert chunks[0].start_index == 0


def test_chunk_size_of_one_sample():
    source = _source(np.arange(5), chunk_size=1)

    chunks = list(source.iter_chunks())

    assert len(chunks) == 5
    assert [chunk.start_index for chunk in chunks] == [0, 1, 2, 3, 4]
    assert all(chunk.num_samples == 1 for chunk in chunks)


# ---------------------------------------------------------------------
#                            Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError):
        _source(np.arange(10), chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [2.5, True])
def test_non_integer_chunk_size_is_rejected(chunk_size):
    with pytest.raises(TypeError):
        _source(np.arange(10), chunk_size=chunk_size)


def test_invalid_replay_mode_is_rejected():
    with pytest.raises(ValueError):
        _source(np.arange(10), mode="turbo")


def test_replay_mode_accepts_a_valid_string():
    source = _source(np.arange(10), mode="real_time")

    assert source.mode is ReplayMode.REAL_TIME


def test_empty_signal_is_rejected():
    with pytest.raises(ValueError):
        _source(np.array([]))


def test_two_dimensional_signal_is_rejected():
    with pytest.raises(ValueError):
        _source(np.zeros((2, 10)))


# ---------------------------------------------------------------------
#                       Real-Time Scheduling
# ---------------------------------------------------------------------


def test_accelerated_replay_never_sleeps():
    clock = FakeClock()
    source = _source(np.arange(12), chunk_size=4, clock=clock)

    list(source.iter_chunks())

    assert clock.sleeps == []


def test_real_time_replay_paces_one_sleep_per_chunk():
    clock = FakeClock()
    # 4 samples at 4 Hz is exactly one second of ECG per chunk.
    source = ReplaySource(
        signal=np.arange(12, dtype=np.float64),
        sampling_rate=4.0,
        chunk_size=4,
        mode=ReplayMode.REAL_TIME,
        clock=clock,
        sleep=clock.sleep,
    )

    chunks = list(source.iter_chunks())

    # One sleep per chunk, never one per sample.
    assert len(chunks) == 3
    assert clock.sleeps == pytest.approx([1.0, 1.0, 1.0])


def test_real_time_replay_uses_absolute_target_times():
    clock = FakeClock()
    source = ReplaySource(
        signal=np.arange(12, dtype=np.float64),
        sampling_rate=4.0,
        chunk_size=4,
        mode=ReplayMode.REAL_TIME,
        clock=clock,
        sleep=clock.sleep,
    )

    chunk_iterator = source.iter_chunks()

    next(chunk_iterator)
    assert clock.sleeps == pytest.approx([1.0])

    # Simulate the consumer spending 0.25 s processing the first chunk.
    # Because chunks are scheduled against absolute targets, the next
    # sleep absorbs that delay instead of drifting by a fixed interval.
    clock.advance(0.25)
    next(chunk_iterator)
    assert clock.sleeps[-1] == pytest.approx(0.75)

    clock.advance(0.10)
    next(chunk_iterator)
    assert clock.sleeps[-1] == pytest.approx(0.90)

    # Total simulated time still matches three seconds of ECG.
    assert clock.now == pytest.approx(1003.0)


def test_real_time_replay_does_not_sleep_when_already_behind():
    clock = FakeClock()
    source = ReplaySource(
        signal=np.arange(12, dtype=np.float64),
        sampling_rate=4.0,
        chunk_size=4,
        mode=ReplayMode.REAL_TIME,
        clock=clock,
        sleep=clock.sleep,
    )

    chunk_iterator = source.iter_chunks()

    next(chunk_iterator)
    sleeps_after_first = len(clock.sleeps)

    # The consumer falls two seconds behind, so the next chunk is overdue
    # and must be delivered immediately rather than sleeping a negative
    # duration.
    clock.advance(2.0)
    next(chunk_iterator)

    assert len(clock.sleeps) == sleeps_after_first


def test_real_time_replay_returns_the_same_samples_as_accelerated():
    signal = np.arange(10, dtype=np.float64)

    accelerated = _source(signal, chunk_size=4)
    real_time_clock = FakeClock()
    real_time = _source(
        signal,
        chunk_size=4,
        mode=ReplayMode.REAL_TIME,
        clock=real_time_clock,
    )

    accelerated_samples = np.concatenate(
        [chunk.samples for chunk in accelerated.iter_chunks()]
    )
    real_time_samples = np.concatenate(
        [chunk.samples for chunk in real_time.iter_chunks()]
    )

    np.testing.assert_array_equal(accelerated_samples, real_time_samples)
    np.testing.assert_array_equal(real_time_samples, signal)
