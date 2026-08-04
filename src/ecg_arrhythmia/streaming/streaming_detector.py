from dataclasses import dataclass

import numpy as np

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.detection.xqrs_detector import XQRSDetector
from ecg_arrhythmia.streaming.sample_buffer import IndexedSampleBuffer


@dataclass(frozen=True)
class DetectorTiming:
    """
    Timing contract for causal detection, in seconds.

    Every streaming detection constant lives here so the behaviour can be
    retuned in one place rather than across the pipeline.
    """

    # Default behaviour:
    # XQRS will analyse up to the most recent 30 seconds
    # of ECG each time it runs.
    analysis_window_seconds: float = 30.0
    # After an analysis, another 5 seconds of new ECG must
    # arrive before XQRS runs again.
    stride_seconds: float = 5.0
    # The detector waits until at least 10 seconds of ECG are
    # available before its first analysis.
    warmup_seconds: float = 10.0
    # Peaks found in the newest 2 seconds are temporarily held
    # back until more ECG arrives, giving XQRS time to settle
    # its decision.
    confirmation_seconds: float = 2.0


class StreamingXQRS:
    """
    Run a whole-signal R-peak detector causally over overlapping windows.

    Running a whole-signal detector on individual chunks would destroy its
    adaptive thresholds, so the detector is instead re-run on the newest
    analysis window once every stride. A peak is released only after the
    confirmation horizon of signal follows it, which gives the detector's
    refractory and backsearch decisions room to settle before they leave
    this class. Peaks are released exactly once, in increasing order.
    """

    def __init__(
        self,
        detector: RPeakDetector | None = None,
        timing: DetectorTiming | None = None,
    ) -> None:
        self._detector = detector if detector is not None else XQRSDetector(learn=True)
        self._timing = timing if timing is not None else DetectorTiming()
        self.reset()

    def reset(self) -> None:
        """Forget every analysis and release decision made so far."""

        # The absolute ECG stop_index when XQRS last ran. It is used
        # to determine whether another 5-second stride has arrived.
        # Set to None for each record.
        self._last_analysis_stop: int | None = None
        # The absolute ECG index of the newest R-peak already returned.
        # Because analysis windows overlap, this prevents the same peak
        # being released more than once.
        self._last_released_peak: int | None = None

    def history_start(self, stop_index: int, sampling_rate: float) -> int:
        """Oldest absolute sample the next analysis window can still need."""

        window = _to_samples(self._timing.analysis_window_seconds, sampling_rate)

        return max(0, stop_index - window)

    def confirmed_peaks(
        self,
        buffer: IndexedSampleBuffer,
        sampling_rate: float,
        end_of_record: bool = False,
    ) -> list[int]:
        """
        Absolute indices of the peaks confirmed since the previous call.

        Returns an empty list before warm-up and between strides. At the
        end of a finite record the stride and the confirmation horizon are
        both dropped so the tail of the record is not withheld.
        """

        # This is the number of samples needed for the warmup window
        warmup = _to_samples(self._timing.warmup_seconds, sampling_rate)

        # If the range of ECG values is less than warmup, we cannot
        # run the detector yet, so we have no peaks
        if buffer.num_retained < warmup:
            return []

        # The stop index is the end of the ECG range we currently have
        # stored
        stop_index = buffer.stop_index

        # If we are not at the end of the record and we have ran
        # analysis
        if not end_of_record and self._last_analysis_stop is not None:
            # convert stride to samples. This is how many samples must
            # atleast pass before we do another analysis.
            stride = _to_samples(self._timing.stride_seconds, sampling_rate)
            # if the number of samples between the current stop index we are
            # holding in the buffer and the last stop index we ran is less than
            # stide (i.e., 5 seconds worth of samples has not passed), we
            # return nothing
            if stop_index - self._last_analysis_stop < stride:
                return []

        # The end of the range is the last analysis stop.
        self._last_analysis_stop = stop_index

        # Number of samples needed for our analysis window.
        window = _to_samples(self._timing.analysis_window_seconds, sampling_rate)

        # We choose stop_index - window in the case our current ECG range holds
        # more samples than window, else we choose the start of the range
        start_index = max(buffer.start_index, stop_index - window)

        # The buffer hands out read-only windows, detectors are third-party
        # code, so give them a writable working copy.
        segment = np.array(buffer.get(start_index, stop_index))

        # This will detect the peaks within the segment. Within the range
        # of ECG values we extracted from the buffer
        peaks = self._detector.detect(signal=segment, sampling_rate=sampling_rate)

        # Because self._detector.detect() returns peak indices relative to the
        # extracted segment, not relative to the original ECG record, we move them
        # into our range by adding the start of our range, start index
        peaks = np.asarray(peaks, dtype=np.int64) + start_index

        # The horizon is how many record we keep for context for the detector
        # to make its decisions. Say we have ECG range [7000, 10000), then
        # with a 2 second confirmation and 360HZ we have a horizon of 2 * 360
        # = 720 samples. So we would retain 720 samples. if we're at the end
        # of the record, no more samples are going to come, so all remaining
        # detected peaks can be considered, so horizon is 0.
        horizon = (
            0
            if end_of_record
            else _to_samples(self._timing.confirmation_seconds, sampling_rate)
        )
        # we then only keep the peaks that are <= 10000 - 720 = 9280
        # We do this because XQRS may change its decision after more
        # ECG arrives because of its adaptive thresholds, refractory
        # checks, and backsearch logic. So the detector waits until
        # a peak has at least two seconds of signal after it before
        # releasing it.
        confirmed = peaks[peaks <= stop_index - horizon]

        # This stops the detector from returning duplicate or previous found
        # peaks. i.e., Suppose the last released peak was 5000, and the new
        # and the new overlapping analysis confirms gives
        # [4800, 5000, 5320, 5650], then we keep only 5320, 5650
        if self._last_released_peak is not None:
            confirmed = confirmed[confirmed > self._last_released_peak]

        # We then set the last released peak equal to the last confirmed
        # peak on this analysis, so 5650 in this example
        if confirmed.size > 0:
            self._last_released_peak = int(confirmed[-1])

        # We then return each confirmed peak
        return [int(peak) for peak in confirmed]


def _to_samples(seconds: float, sampling_rate: float) -> int:
    """Convert a timing constant into whole samples."""

    return int(round(seconds * sampling_rate))
