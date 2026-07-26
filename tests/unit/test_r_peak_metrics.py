import numpy as np
import pytest

from ecg_arrhythmia.evaluation.r_peak_matching import match_r_peaks
from ecg_arrhythmia.evaluation.r_peak_metrics import (
    compute_r_peak_metrics,
)


def test_compute_r_peak_metrics_returns_expected_detection_metrics():
    match_result = match_r_peaks(
        annotation_samples=np.array([100, 300, 500]),
        detected_samples=np.array([105, 495, 700]),
        tolerance_samples=10,
    )

    metrics = compute_r_peak_metrics(
        match_result=match_result,
        sampling_rate=360,
    )

    # Matches:
    # 100 ↔ 105
    # 500 ↔ 495
    #
    # Unmatched annotation: 300
    # Unmatched detection: 700
    assert metrics.true_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 1

    assert metrics.num_annotations == 3
    assert metrics.num_detections == 3

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_compute_r_peak_metrics_returns_expected_offset_statistics():
    match_result = match_r_peaks(
        annotation_samples=np.array([100, 300]),
        detected_samples=np.array([95, 307]),
        tolerance_samples=10,
    )

    metrics = compute_r_peak_metrics(
        match_result=match_result,
        sampling_rate=360,
    )

    # Signed offsets are -5 and +7 samples.
    assert metrics.mean_offset_samples == pytest.approx(1.0)
    assert metrics.mean_absolute_offset_samples == pytest.approx(6.0)
    assert metrics.median_absolute_offset_samples == pytest.approx(6.0)
    assert metrics.maximum_absolute_offset_samples == pytest.approx(7.0)

    # At 360 Hz, one sample is approximately 2.778 ms.
    assert metrics.mean_offset_ms == pytest.approx(1000 / 360)
    assert metrics.mean_absolute_offset_ms == pytest.approx(6 * 1000 / 360)


def test_compute_r_peak_metrics_handles_no_detections():
    match_result = match_r_peaks(
        annotation_samples=np.array([100, 300]),
        detected_samples=np.array([], dtype=np.int64),
        tolerance_samples=10,
    )

    metrics = compute_r_peak_metrics(
        match_result=match_result,
        sampling_rate=360,
    )

    assert metrics.true_positives == 0
    assert metrics.false_negatives == 2
    assert metrics.false_positives == 0

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0

    assert metrics.mean_offset_samples is None
    assert metrics.mean_absolute_offset_ms is None


def test_compute_r_peak_metrics_handles_no_annotations_or_detections():
    match_result = match_r_peaks(
        annotation_samples=np.array([], dtype=np.int64),
        detected_samples=np.array([], dtype=np.int64),
        tolerance_samples=10,
    )

    metrics = compute_r_peak_metrics(
        match_result=match_result,
        sampling_rate=360,
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


@pytest.mark.parametrize(
    "sampling_rate",
    [0, -360, np.nan, np.inf],
)
def test_compute_r_peak_metrics_rejects_invalid_sampling_rate(
    sampling_rate,
):
    match_result = match_r_peaks(
        annotation_samples=np.array([100]),
        detected_samples=np.array([100]),
        tolerance_samples=10,
    )

    with pytest.raises(ValueError):
        compute_r_peak_metrics(
            match_result=match_result,
            sampling_rate=sampling_rate,
        )
