import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.streaming.sample_buffer import IndexedSampleBuffer
from ecg_arrhythmia.streaming.streaming_detector import (
    DetectorTiming,
    StreamingXQRS,
)

# One sample per second keeps every timing constant readable as samples.
SAMPLING_RATE = 1.0
TIMING = DetectorTiming(
    analysis_window_seconds=30.0,
    stride_seconds=5.0,
    warmup_seconds=10.0,
    confirmation_seconds=2.0,
)


class MarkerDetector(RPeakDetector):
    """Fake detector that treats every sample equal to 1.0 as an R-peak."""

    @property
    def name(self) -> str:
        return "marker"

    def _detect(
        self,
        signal: NDArray[np.float64],
        sampling_rate: float,
    ) -> NDArray[np.int64]:
        return np.flatnonzero(signal == 1.0).astype(np.int64)


def _record(length: int, markers: list[int]) -> NDArray[np.float64]:
    signal = np.zeros(length, dtype=np.float64)
    signal[markers] = 1.0
    return signal


def _buffer(signal: NDArray[np.float64], stop_index: int) -> IndexedSampleBuffer:
    buffer = IndexedSampleBuffer()
    buffer.append(signal[:stop_index], 0)
    return buffer


def _detector() -> StreamingXQRS:
    return StreamingXQRS(detector=MarkerDetector(), timing=TIMING)


def test_no_detection_before_warm_up():
    signal = _record(40, markers=[3])

    assert _detector().confirmed_peaks(_buffer(signal, 9), SAMPLING_RATE) == []


def test_analysis_only_runs_once_the_stride_is_reached():
    signal = _record(40, markers=[3, 13])
    detector = _detector()
    buffer = _buffer(signal, 10)

    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == [3]

    # Four more samples is short of the five-sample stride.
    buffer.append(signal[10:14], 10)
    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == []

    buffer.append(signal[14:16], 14)
    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == [13]


def test_peaks_are_translated_from_the_window_to_absolute_indices():
    signal = _record(40, markers=[22])
    buffer = _buffer(signal, 30)

    # The window starts at sample 5, so the detector sees the peak at 17.
    buffer.prune_before(5)
    assert buffer.start_index == 5

    assert _detector().confirmed_peaks(buffer, SAMPLING_RATE) == [22]


def test_a_peak_inside_the_confirmation_horizon_is_withheld():
    signal = _record(40, markers=[9])
    detector = _detector()
    buffer = _buffer(signal, 10)

    # Only one sample follows the peak, short of the two-sample horizon.
    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == []

    buffer.append(signal[10:15], 10)
    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == [9]


def test_overlapping_passes_release_each_peak_once():
    signal = _record(40, markers=[3, 13, 23])
    detector = _detector()
    buffer = _buffer(signal, 10)

    released = []
    for stop_index in range(15, 41, 5):
        released.extend(detector.confirmed_peaks(buffer, SAMPLING_RATE))
        buffer.append(signal[buffer.stop_index : stop_index], buffer.stop_index)
    released.extend(detector.confirmed_peaks(buffer, SAMPLING_RATE))

    assert released == [3, 13, 23]


def test_flush_releases_the_tail_of_a_finite_record():
    signal = _record(40, markers=[13, 39])
    detector = _detector()
    buffer = _buffer(signal, 20)

    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == [13]

    buffer.append(signal[20:40], 20)

    # The final peak sits inside the confirmation horizon until the
    # record is known to have ended.
    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == []
    assert detector.confirmed_peaks(
        buffer,
        SAMPLING_RATE,
        end_of_record=True,
    ) == [39]


def test_reset_forgets_previously_released_peaks():
    signal = _record(40, markers=[3])
    detector = _detector()
    buffer = _buffer(signal, 10)

    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == [3]

    detector.reset()

    assert detector.confirmed_peaks(buffer, SAMPLING_RATE) == [3]


def test_history_start_keeps_one_analysis_window():
    assert _detector().history_start(stop_index=100, sampling_rate=SAMPLING_RATE) == 70
    assert _detector().history_start(stop_index=10, sampling_rate=SAMPLING_RATE) == 0
