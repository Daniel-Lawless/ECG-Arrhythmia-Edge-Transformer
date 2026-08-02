import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.streaming.sample_chunk import validate_start_index

# Starting capacity in samples, grown by doubling as chunks arrive.
INITIAL_CAPACITY = 1024


class IndexedSampleBuffer:
    """
    Rolling ECG history addressed by absolute record-relative index.

    Invariant: the samples for absolute indices [start_index, stop_index)
    are retained, and absolute index i lives in storage at
    _offset + (i - start_index). Appending only raises stop_index and
    pruning only raises start_index, so a retained index never changes
    its absolute position. Pruning is a bookkeeping change; the storage is
    compacted lazily, only when an append would otherwise not fit.
    """

    def __init__(self, start_index: int = 0) -> None:
        self.reset(start_index)

    def reset(self, start_index: int = 0) -> None:
        """Discard all history and rebase on a new record start."""

        self._samples: NDArray[np.float64] = np.empty(
            INITIAL_CAPACITY,
            dtype=np.float64,
        )
        self._offset = 0
        self._length = 0
        self._start_index = validate_start_index(start_index)

    @property
    def start_index(self) -> int:
        """Absolute index of the oldest retained sample."""

        return self._start_index

    @property
    def stop_index(self) -> int:
        """
        Absolute index one past the newest appended sample.
        This should be the start index of the next chunk
        """

        return self._start_index + self._length

    @property
    def num_retained(self) -> int:
        """Number of samples currently held in memory."""

        return self._length

    def append(self, samples: NDArray[np.floating], start_index: int) -> None:
        """Append the next contiguous block of samples."""

        # If the start index of the chunk is not equal to the stop index,
        # then it is not continuous, an error should be raised.
        if start_index != self.stop_index:
            raise ValueError(
                f"Samples must continue the buffer at {self.stop_index}, but "
                f"the block starts at {start_index}."
            )

        # we convert the samples to a numpy array. We expect this, but we can
        # do it for completeness.
        block = np.asarray(samples, dtype=np.float64)

        if block.ndim != 1:
            raise ValueError(
                f"Samples must be one-dimensional, received shape {block.shape}."
            )

        if self._offset + self._length + block.size > self._samples.size:
            self._compact(block.size)

        write_from = self._offset + self._length
        self._samples[write_from : write_from + block.size] = block
        self._length += block.size

    def get(self, start_index: int, stop_index: int) -> NDArray[np.float64]:
        """
        Return the samples for [start_index, stop_index).

        The result is a read-only copy, so it stays valid after later
        appends or compaction move the underlying storage.
        """

        if stop_index < start_index:
            raise ValueError(
                f"Stop index {stop_index} must not precede start index {start_index}."
            )

        if start_index < self._start_index or stop_index > self.stop_index:
            raise ValueError(
                f"Samples [{start_index}, {stop_index}) are outside the "
                f"retained history [{self.start_index}, {self.stop_index})."
            )

        first = self._offset + (start_index - self._start_index)
        window = self._samples[first : first + (stop_index - start_index)].copy()
        window.setflags(write=False)

        return window

    def prune_before(self, index: int) -> None:
        """
        Discard retained history older than index.

        Requests below the current start or beyond the newest sample are
        clamped, so a caller can never discard samples it has not seen.
        """

        keep_from = min(max(index, self._start_index), self.stop_index)
        discarded = keep_from - self._start_index

        if discarded <= 0:
            return

        self._offset += discarded
        self._length -= discarded
        self._start_index = keep_from

    def _compact(self, incoming: int) -> None:
        """Move retained samples to the front, growing storage if needed."""

        retained = self._samples[self._offset : self._offset + self._length]
        required = self._length + incoming

        if required > self._samples.size:
            grown = np.empty(
                max(required, 2 * self._samples.size),
                dtype=np.float64,
            )
            grown[: self._length] = retained
            self._samples = grown
        else:
            self._samples[: self._length] = retained.copy()

        self._offset = 0
