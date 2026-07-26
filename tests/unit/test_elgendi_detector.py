from unittest.mock import patch

import numpy as np

from ecg_arrhythmia.detection.elgendi_detector import ElgendiDetector

# The detector imports these functions directly into the shared base
# module, so the tests patch the names where the detector looks them up.
CLEAN_TARGET = "ecg_arrhythmia.detection.neurokit_detector.ecg_clean"

PEAKS_TARGET = "ecg_arrhythmia.detection.neurokit_detector.ecg_peaks"


def test_elgendi_detector_has_expected_name():
    assert ElgendiDetector().name == "elgendi"


# The decorators apply bottom-up, so `mock_ecg_clean` corresponds to the
# inner ecg_clean patch and `mock_ecg_peaks` to the outer ecg_peaks patch.
@patch(PEAKS_TARGET)
@patch(CLEAN_TARGET)
def test_elgendi_detector_forwards_signal_and_uses_correct_method(
    mock_ecg_clean,
    mock_ecg_peaks,
):
    cleaned_signal = np.arange(600, dtype=np.float64)
    mock_ecg_clean.return_value = cleaned_signal
    mock_ecg_peaks.return_value = (
        None,
        {"ECG_R_Peaks": np.array([120, 360, 540], dtype=np.int64)},
    )

    detector = ElgendiDetector()
    signal = np.zeros(600)

    peak_indices = detector.detect(
        signal=signal,
        sampling_rate=360,
    )

    np.testing.assert_array_equal(
        peak_indices,
        np.array([120, 360, 540]),
    )

    # The raw signal is cleaned exactly once with Elgendi's method.
    mock_ecg_clean.assert_called_once()
    clean_call = mock_ecg_clean.call_args
    np.testing.assert_array_equal(clean_call.args[0], signal)
    assert clean_call.kwargs["sampling_rate"] == 360
    assert clean_call.kwargs["method"] == "elgendi2010"

    # Peak detection runs once on the cleaned signal with Elgendi's method.
    mock_ecg_peaks.assert_called_once()
    peaks_call = mock_ecg_peaks.call_args
    np.testing.assert_array_equal(peaks_call.args[0], cleaned_signal)
    assert peaks_call.kwargs["sampling_rate"] == 360
    assert peaks_call.kwargs["method"] == "elgendi2010"


@patch(PEAKS_TARGET)
@patch(CLEAN_TARGET)
def test_elgendi_detector_cleans_signal_exactly_once(
    mock_ecg_clean,
    mock_ecg_peaks,
):
    mock_ecg_clean.return_value = np.zeros(500, dtype=np.float64)
    mock_ecg_peaks.return_value = (
        None,
        {"ECG_R_Peaks": np.array([50, 200], dtype=np.int64)},
    )

    ElgendiDetector().detect(
        signal=np.zeros(500),
        sampling_rate=360,
    )

    assert mock_ecg_clean.call_count == 1
    assert mock_ecg_peaks.call_count == 1


@patch(PEAKS_TARGET)
@patch(CLEAN_TARGET)
def test_elgendi_detector_allows_no_detected_peaks(
    mock_ecg_clean,
    mock_ecg_peaks,
):
    mock_ecg_clean.return_value = np.zeros(400, dtype=np.float64)
    mock_ecg_peaks.return_value = (
        None,
        {"ECG_R_Peaks": np.array([], dtype=np.int64)},
    )

    peak_indices = ElgendiDetector().detect(
        signal=np.zeros(400),
        sampling_rate=360,
    )

    assert peak_indices.shape == (0,)
    assert peak_indices.dtype == np.int64


@patch(PEAKS_TARGET)
@patch(CLEAN_TARGET)
def test_elgendi_detector_returns_int64_peak_indices(
    mock_ecg_clean,
    mock_ecg_peaks,
):
    mock_ecg_clean.return_value = np.zeros(400, dtype=np.float64)
    mock_ecg_peaks.return_value = (
        None,
        {"ECG_R_Peaks": np.array([100, 300], dtype=np.int32)},
    )

    peak_indices = ElgendiDetector().detect(
        signal=np.zeros(400),
        sampling_rate=360,
    )

    assert peak_indices.dtype == np.int64
