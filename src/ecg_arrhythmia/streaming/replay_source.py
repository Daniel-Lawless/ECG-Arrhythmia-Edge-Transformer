import logging
import time
from collections.abc import Callable, Iterator
from enum import StrEnum
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.streaming.sample_chunk import (
    SampleChunk,
    validate_samples,
    validate_sampling_rate,
)

logger = logging.getLogger(__name__)

# 36 samples is 100 ms at 360 Hz, which is a realistic acquisition block
# for an edge device without making the chunk count unnecessarily large.
DEFAULT_CHUNK_SIZE = 36

# Injectable timing dependencies. Both default to the standard library but
# are replaced by fakes in tests so no test has to wait in real time.
Clock = Callable[[], float]
SleepFunction = Callable[[float], None]


class ReplayMode(StrEnum):
    """How quickly a record is replayed into the streaming engine."""

    # Emit chunks as fast as the consumer can accept them.
    ACCELERATED = "accelerated"

    # Emit chunks at the pace the signal was originally sampled.
    REAL_TIME = "real_time"


class ReplaySource:
    """
    Replay one ECG record as consecutive `SampleChunk` objects.

    The source only reads and slices the signal: sample values are never
    altered, reordered or resampled. It is deliberately independent of the
    streaming engine, so a future Raspberry Pi or device source can emit
    the same chunk representation into the same engine.
    """

    def __init__(
        self,
        signal: NDArray[np.floating],
        sampling_rate: float,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        mode: ReplayMode | str = ReplayMode.ACCELERATED,
        record_name: str | None = None,
        lead_name: str | None = None,
        clock: Clock = perf_counter,
        sleep: SleepFunction = time.sleep,
    ) -> None:

        # ReplayMode(...) rejects any unknown mode string with a ValueError.
        self.mode = ReplayMode(mode)

        self.signal = validate_samples(signal)
        self.sampling_rate = validate_sampling_rate(sampling_rate)
        self.chunk_size = validate_chunk_size(chunk_size)
        self.record_name = record_name
        self.lead_name = lead_name

        self._clock = clock
        self._sleep = sleep

    @classmethod
    def from_record(
        cls,
        record_name: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        mode: ReplayMode | str = ReplayMode.ACCELERATED,
        clock: Clock = perf_counter,
        sleep: SleepFunction = time.sleep,
    ) -> "ReplaySource":
        """
        This builds a replay souce from a MIT-BIH record. We return
        a ReplaySource object, which will run the __init__() constructor
        which will validation the ecg record samples, sampling rate,
        and chunk_size.
        """

        signals, fields, _ = load_record(record_name=record_name)
        signal, lead_name = select_signal_channel(signals=signals, fields=fields)

        return cls(
            signal=signal,
            sampling_rate=float(fields["fs"]),
            chunk_size=chunk_size,
            mode=mode,
            record_name=record_name,
            lead_name=lead_name,
            clock=clock,
            sleep=sleep,
        )

    @property
    def num_samples(self) -> int:
        """Total number of samples in the record being replayed."""

        return int(self.signal.size)

    @property
    def num_chunks(self) -> int:
        """
        Number of chunks that fit into this signal, including a
        partial one.
        """

        return -(-self.num_samples // self.chunk_size)

    # This is an important function. It introduces several important
    # Python ideas at once. Iterator[SampleChunk] tells us this function
    # can provide SampleChunk objects one after another sequentially.

    # When we do chunk_iterator = source.iter_chunks(), this does not run
    # this function, it creates a generator object. It runs when we do
    # next(chunk_iterator) or for chunk in chunk_iterator (Python calls
    # next() when we use a for loop)
    def iter_chunks(self, on_schedule=None) -> Iterator[SampleChunk]:
        """
        Yield the record as consecutive chunks in sample order.

        In real-time mode each chunk is scheduled against an absolute
        target time derived from the replay start, so pacing errors do not
        accumulate across a long record. Sleeping happens once per chunk
        rather than once per sample.
        """

        # If mode is real time, true, else false.
        real_time = self.mode is ReplayMode.REAL_TIME
        # If it is real time, start the clock, else no timing is needed.
        start_time = self._clock() if real_time else 0.0

        # Starting at the start of the ecg signal, and moving chunk_size
        # through it
        for start_index in range(0, self.num_samples, self.chunk_size):
            # calculate the stop index. If the start index + chunk_size
            # is greater than the num aplitude values, then chose num_samples
            # as the stop index.
            stop_index = min(start_index + self.chunk_size, self.num_samples)

            # Create the chunk. This will validate and return a SampleChunk
            # object
            chunk = SampleChunk(
                samples=self.signal[start_index:stop_index],
                start_index=start_index,
                sampling_rate=self.sampling_rate,
            )

            # If we're in real_time mode
            if real_time:
                # target_time is the time at which wew should have delieved a
                # certain number of chunks. i.e., for two chunks,
                # stop_index = 72, so 72 / 360 = 0.2s so by start_time + 0.2
                # we should have delieved 2 chunks
                target_time = start_time + (stop_index / self.sampling_rate)
                # if the time it has taken so far is more than target time,
                # we are behind schedule, in which case we deliever (yeild)
                # the chunk straight away to try to catch up
                remaining_seconds = target_time - self._clock()

                # if the time it has taken so far is less than the target time,
                # we are ahead of schedule, so we sleep the remaining seconds
                # until we deliever the chunk to stay at real time.
                if remaining_seconds > 0:
                    self._sleep(remaining_seconds)

            if on_schedule is not None:
                on_schedule(chunk, target_time if real_time else None)

            yield chunk


def validate_chunk_size(chunk_size: int):

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("Chunk size must be an integer number of samples.")

    if chunk_size < 1:
        raise ValueError("Chunk size must be at least one sample.")

    return chunk_size
