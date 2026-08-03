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
    its absolute position. Pruning is a bookkeeping change, the storage is
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
        """
        This and stop_index refer to the ECG values range that
        we're currently holding, not the buffer indices.
        Start index refers to the start of the range, i.e.,
        [400, 700)
        """

        return self._start_index

    @property
    def stop_index(self) -> int:
        """
        Stop index refers to the end of the ECG values range
        that we're currently holding. so in the previous examples
        [400, 700), start_index = 400, stop_index = 700. stop index
        should be equal to the start index of the next chunk
        """

        return self._start_index + self._length

    @property
    def num_retained(self) -> int:
        """
        Number of samples currently held in memory. Essentially
        how large the range of ECG values we currently have is.
        i.e., for [400, 700) is 300 samples
        """

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

        # Offset refers to where that range is held in the buffer.
        # if the offest + the size of the range + the incoming chunk samples
        # goes beyond the capacity of the numpy array, we make it bigger.
        if self._offset + self._length + block.size > self._samples.size:
            self._compact(block.size)

        # if we have enough space, or just created enough space, we get the index
        # to write the new ECG samples to in the buffer. say we have range [400, 700)
        # and buffer [200, 500), then we start at 200, add the range 300, which gives us
        # index 500.
        write_from = self._offset + self._length
        # We then write the values there.
        self._samples[write_from : write_from + block.size] = block
        # The length of samples stored has then increased by that number of samples
        self._length += block.size

    def get(self, start_index: int, stop_index: int) -> NDArray[np.float64]:
        """
        Return the samples for [start_index, stop_index). This is asking
        for a certain range of ECG values

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

        # We find the start index of the requested ECG range in the buffer
        first = self._offset + (start_index - self._start_index)
        # We then extract the ECG samples from that start index + the range of the
        # requested ECG samples and make a copy of it.
        window = self._samples[first : first + (stop_index - start_index)].copy()
        # Make the copy read-only
        window.setflags(write=False)

        # Return the requested ECG sample range
        return window

    def prune_before(self, index: int) -> None:
        """
        Discard retained history older than index.

        Requests below the current start or beyond the newest sample are
        clamped, so a caller can never discard samples it has not seen.
        """

        # This ensures the index is within the ECG range.
        keep_from = min(max(index, self._start_index), self.stop_index)
        # We want to discard all samples between keep from and the start index
        discarded = keep_from - self._start_index

        # This means keep_from = start index, so we keep the same range
        if discarded == 0:
            return

        # [400, 700), index 450, offset = 200 [200, 500).
        # new range [450, 700), offset must move up 50.
        self._offset += discarded
        # The range has decreased by 50
        self._length -= discarded
        # the start index is now keep_from to stop_index
        self._start_index = keep_from

    def _compact(self, incoming: int) -> None:
        """Move retained samples to the front, growing storage if needed."""

        # retained is the buffer positions for the ECG range we are currently
        # holding
        retained = self._samples[self._offset : self._offset + self._length]
        # Required is the range we curently have + the number of chunk samples
        # being appended.
        required = self._length + incoming

        # required is greater than sample size if the size of the range
        # + the incoming samples is greater than the capacity.
        if required > self._samples.size:
            # In which case we create a new empty array and make it as big
            # as required or double its size, depending on which is bigger.
            grown = np.empty(
                max(required, 2 * self._samples.size),
                dtype=np.float64,
            )
            # We then add the retained range into this new array
            grown[: self._length] = retained
            # set our object buffer to grown
            self._samples = grown
        else:
            # Else we move the ECG range to the start of the array
            self._samples[: self._length] = retained.copy()

        # Both cases result in the offset being 0
        self._offset = 0
