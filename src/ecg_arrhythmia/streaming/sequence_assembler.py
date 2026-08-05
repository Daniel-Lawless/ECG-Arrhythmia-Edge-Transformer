from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.preprocessing.beat_extraction import (
    LOCAL_RR_WINDOW,
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    SEQUENCE_LENGTH,
    previous_rr_seconds,
    rr_ratio,
)
from ecg_arrhythmia.streaming.sample_buffer import IndexedSampleBuffer


@dataclass(frozen=True)
class BeatSequence:
    """
    One model-ready input built from consecutive detected beats.

    ecg has shape (sequence_length, 1, 240) and rr has shape
    (sequence_length, 2), holding previous_rr_seconds and rr_ratio.
    peak_indices records the absolute R-peak behind each beat, which the
    parity evaluator uses to explain differences without the pipeline
    having to retain any extra history. target_beat_index is the index
    of the last beat in the sequence, the beat we're classifying.
    """

    ecg: NDArray[np.float64]
    rr: NDArray[np.float64]
    target_peak_index: int
    peak_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Beat:
    # The beats peak index
    peak_index: int
    # Its 240 ECG sample window
    window: NDArray[np.float64]
    # Its two rr features, prev_rr and rr_ratio
    rr: tuple[float, float]


class SequenceAssembler:
    """
    Turn confirmed R-peaks into model-ready beat sequences.

    Beat and RR semantics match the offline XQRS-centred builder exactly:
    every confirmed peak advances the RR chain, a peak only becomes a beat
    once its complete 240-sample window exists, and only completed beats
    contribute to the local RR history. Unmatched detections therefore
    still shape the RR features and the sequence context, which is what
    makes streaming output comparable with the offline dataset.
    """

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH) -> None:
        if sequence_length < 1:
            raise ValueError("Sequence length must be at least one beat.")

        self._sequence_length = sequence_length
        self.reset()

    def reset(self, record_start: int = 0) -> None:
        """Discard all beat, RR and pending-peak state for a new record."""

        # Absolute starting index of the record.
        self._record_start = record_start
        # Holds confirmed R-peaks that have not yet been turned
        # into beats
        self._pending: deque[int] = deque()
        # Stores the previous r-peak so the prev_rr can be calculated
        self._previous_peak: int | None = None
        # Stores up to the latest 10 prev_rr intervals for calculating rr_ratio.
        self._recent_rr: deque[float] = deque(maxlen=LOCAL_RR_WINDOW)
        # Stores the most recent completed beats. With sequence length 5,
        # it automatically keeps only the latest 5 beats.
        self._beats: deque[_Beat] = deque(maxlen=self._sequence_length)

    @property
    def history_start(self) -> int | None:
        """Oldest sample still needed by a peak awaiting its beat window."""

        # If there are not peaks waiting to become beats, then return None
        if not self._pending:
            return None

        # Say the oldest peak is at index 1000, the beat window for
        # that peak needs ECG samples beginning at absolute index
        # 910, so the buffer must not prune anything from 910
        # onward yet.
        return self._pending[0] - SAMPLES_BEFORE

    def add_peaks(self, peak_indices: Iterable[int]) -> None:
        """Queue newly confirmed peaks in detection order."""

        # Adds the peaks we just detected into the pending queue
        self._pending.extend(int(peak) for peak in peak_indices)

    def drain(
        self,
        buffer: IndexedSampleBuffer,
        end_of_record: bool = False,
    ) -> list[BeatSequence]:
        """
        Complete every pending peak whose beat window has fully arrived.

        At the end of a record the remaining peaks are resolved against
        the samples that exist, so a beat truncated by the record boundary
        is dropped rather than waited for.
        """

        sequences: list[BeatSequence] = []

        # While we still have pending peaks
        while self._pending:
            # Start from the oldest peak awaiting sequencing
            peak_index = self._pending[0]

            # If we are not at the end of the record and we do not have the required
            # samples after the peak stored in the buffer, we break
            if not end_of_record and peak_index + SAMPLES_AFTER > buffer.stop_index:
                # The rest of this beat is still to come.
                break

            # Popleft removes the left element, which is the oldest.
            self._pending.popleft()
            # builds the Beat object for this peak
            beat = self._build_beat(peak_index, buffer)

            # If beat is None, i.e., there is no previous peak or the window
            # could not be created, we move onto the next peak
            if beat is None:
                continue

            # We add this beat to out queue of beats
            self._beats.append(beat)

            # If we have SEQUENCE_LENGTH beats, we build a sequence from that.
            if len(self._beats) == self._sequence_length:
                sequences.append(self._build_sequence())

        return sequences

    def _build_beat(
        self,
        peak_index: int,
        buffer: IndexedSampleBuffer,
    ) -> _Beat | None:
        # self._previous_peak refers to the peak index of the last peak,
        # we set that equal to previous peak and set this current peak
        # index equal to self._previous_peak.
        previous_peak = self._previous_peak
        self._previous_peak = peak_index

        # The first detection of a record cannot have an RR interval
        if previous_peak is None:
            return None

        # Calculate the seconds between this peak and the previous peak
        prev_rr = previous_rr_seconds(peak_index, previous_peak)

        # calculate where the window should begin and stop around this peak
        window_start = peak_index - SAMPLES_BEFORE
        window_stop = peak_index + SAMPLES_AFTER

        # A window truncated by either end of the record is skipped, but
        # the RR chain has already advanced, exactly as offline.
        if window_start < self._record_start or window_stop > buffer.stop_index:
            return None

        # Calculate the rr_ratio, telling us how late/early/expected this beat
        # arrived.
        ratio = rr_ratio(prev_rr, self._recent_rr)
        # If the queue is already full (10 RR intervals), it automatically
        # ejects the oldest RR interval from the queue, and puts this
        # at the front. i.e., [1...10] append(11) -> [2...11]
        self._recent_rr.append(prev_rr)

        # Returns a beat object that stores the peak_index, the window
        # surrounding the peak, and the time between this peak and the
        # last and its ratio
        return _Beat(
            peak_index=peak_index,
            window=buffer.get(window_start, window_stop),
            rr=(prev_rr, ratio),
        )

    def _build_sequence(self) -> BeatSequence:
        # We only end up here when we have SEQUENCE_LENGTH beats
        beats = list(self._beats)

        return BeatSequence(
            # np.stack gives us (SEQUENCE_LENGTH, 240), then np.newaxis
            # gives us the extra 1 signal lead we need for
            # (SEQUENCE_LENGTH, 1, 240)
            ecg=np.stack([beat.window for beat in beats])[:, np.newaxis, :],
            # Extract the rr feature for each beat
            rr=np.asarray([beat.rr for beat in beats], dtype=np.float64),
            # the beat index is the peak index for the last beat in the
            # sequence
            target_peak_index=beats[-1].peak_index,
            # peak indices for this sequence is the peak index for each beat
            peak_indices=tuple(beat.peak_index for beat in beats),
        )
