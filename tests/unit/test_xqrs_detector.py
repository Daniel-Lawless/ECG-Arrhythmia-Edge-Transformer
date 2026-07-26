from unittest.mock import patch

import numpy as np

from ecg_arrhythmia.detection.xqrs_detector import XQRSDetector


def test_xqrs_detector_has_expected_name():
    # Create the detector using its default configuration.
    detector = XQRSDetector()

    # Ensure the detector exposes the detectors identifier.
    assert detector.name == "xqrs"


# Temporarily replace WFDB's real xqrs_detect function with a mock.
# This lets us test our wrapper without running the actual XQRS algorithm.
@patch("ecg_arrhythmia.detection.xqrs_detector.processing.xqrs_detect")
def test_xqrs_detector_calls_wfdb_xqrs(mock_xqrs_detect):
    # Configure the mocked WFDB function to return known peak indices.
    mock_xqrs_detect.return_value = np.array(
        [100, 300, 500],
        dtype=np.int64,
    )

    # Create the detector with XQRS parameter learning enabled.
    detector = XQRSDetector(learn=True)

    # Create a dummy 1d 600 sample ECG signal.
    signal = np.zeros(600)

    # Run detection. This should call out _detect function
    peak_indices = detector.detect(
        signal=signal,
        sampling_rate=360,
    )

    # Ensure the detector returns the peak indices produced by our
    # _detect function.
    np.testing.assert_array_equal(
        peak_indices,
        np.array([100, 300, 500]),
    )

    # Retrieve the keyword arguments that our detector passed to
    # the mocked WFDB xqrs_detect function.
    call_arguments = mock_xqrs_detect.call_args.kwargs

    # Ensure the original ECG signal was passed to WFDB as `sig`.
    np.testing.assert_array_equal(
        call_arguments["sig"],
        signal,
    )

    # Ensure the sampling rate was passed to WFDB as `fs`.
    assert call_arguments["fs"] == 360

    # Ensure the detector's learning configuration was forwarded.
    assert call_arguments["learn"] is True

    # Ensure WFDB's console output is disabled.
    assert call_arguments["verbose"] is False


@patch("ecg_arrhythmia.detection.xqrs_detector.processing.xqrs_detect")
def test_xqrs_detector_can_disable_parameter_learning(
    mock_xqrs_detect,
):
    # Configure the mock to return valid peak indices.
    mock_xqrs_detect.return_value = np.array(
        [100, 300],
        dtype=np.int64,
    )

    # Create the detector with parameter learning disabled.
    detector = XQRSDetector(learn=False)

    # Run detection so that the mocked WFDB function is called.
    detector.detect(
        signal=np.zeros(400),
        sampling_rate=360,
    )

    # Retrieve the arguments passed to the mocked WFDB function.
    call_arguments = mock_xqrs_detect.call_args.kwargs

    # Ensure the detector correctly forwards learn=False.
    assert call_arguments["learn"] is False


@patch("ecg_arrhythmia.detection.xqrs_detector.processing.xqrs_detect")
def test_xqrs_detector_returns_int64_peak_indices(
    mock_xqrs_detect,
):
    # Simulate WFDB returning valid integer indices using int32.
    mock_xqrs_detect.return_value = np.array(
        [100, 300],
        dtype=np.int32,
    )

    detector = XQRSDetector()

    peak_indices = detector.detect(
        signal=np.zeros(400),
        sampling_rate=360,
    )

    # Ensure the base detector validation converts all valid
    # integer peak indices to the int64 dtype.
    assert peak_indices.dtype == np.int64


@patch("ecg_arrhythmia.detection.xqrs_detector.processing.xqrs_detect")
def test_xqrs_detector_allows_no_detected_peaks(
    mock_xqrs_detect,
):
    # Simulate WFDB finding no R-peaks in the supplied signal.
    mock_xqrs_detect.return_value = np.array([])

    detector = XQRSDetector()

    peak_indices = detector.detect(
        signal=np.zeros(400),
        sampling_rate=360,
    )

    # An empty result should remain a valid one-dimensional array.
    assert peak_indices.shape == (0,)

    # Even an empty result should satisfy the detector's int64
    # output contract.
    assert peak_indices.dtype == np.int64
