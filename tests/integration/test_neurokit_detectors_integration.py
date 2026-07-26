import numpy as np
import pytest

from ecg_arrhythmia.data.load_record import (
    load_record,
    select_signal_channel,
)
from ecg_arrhythmia.detection.elgendi_detector import ElgendiDetector
from ecg_arrhythmia.detection.hamilton_detector import HamiltonDetector
from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.evaluation.r_peak_matching import match_r_peaks
from ecg_arrhythmia.evaluation.r_peak_metrics import compute_r_peak_metrics
from ecg_arrhythmia.preprocessing.beat_extraction import BEAT_SYMBOLS


@pytest.mark.parametrize(
    "detector",
    [
        HamiltonDetector(),
        ElgendiDetector(),
    ],
)
def test_neurokit_detector_finds_r_peaks_in_real_mit_bih_record(
    detector: RPeakDetector,
):
    # Record 100 is a clean MIT-BIH record suitable for a broad,
    # non-brittle behavioural check of the real algorithms.
    signals, fields, annotation = load_record("100")

    signal, _ = select_signal_channel(
        signals=signals,
        fields=fields,
    )

    sampling_rate = float(fields["fs"])

    annotation_samples = np.array(
        [
            sample
            for sample, symbol in zip(
                annotation.sample,
                annotation.symbol,
                strict=True,
            )
            if symbol in BEAT_SYMBOLS
        ],
        dtype=np.int64,
    )

    detected_samples = detector.detect(
        signal=signal,
        sampling_rate=sampling_rate,
    )

    # The detector output must satisfy the shared contract.
    assert detected_samples.size > 0
    assert detected_samples.ndim == 1
    assert detected_samples.dtype == np.int64
    assert np.all(np.diff(detected_samples) > 0)
    assert detected_samples[0] >= 0
    assert detected_samples[-1] < len(signal)

    tolerance_samples = round(0.1 * sampling_rate)

    match_result = match_r_peaks(
        annotation_samples=annotation_samples,
        detected_samples=detected_samples,
        tolerance_samples=tolerance_samples,
    )

    metrics = compute_r_peak_metrics(
        match_result=match_result,
        sampling_rate=sampling_rate,
    )

    # Use broad thresholds so the test stays stable across NeuroKit
    # versions rather than asserting exact detected positions.
    assert metrics.recall > 0.95
    assert metrics.precision > 0.90
    assert metrics.mean_absolute_offset_ms is not None
    assert metrics.mean_absolute_offset_ms < 100.0
