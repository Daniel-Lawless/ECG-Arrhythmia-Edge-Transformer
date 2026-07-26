import numpy as np

from ecg_arrhythmia.data.load_record import (
    load_record,
    select_signal_channel,
)
from ecg_arrhythmia.detection.xqrs_detector import XQRSDetector
from ecg_arrhythmia.evaluation.r_peak_matching import match_r_peaks
from ecg_arrhythmia.evaluation.r_peak_metrics import compute_r_peak_metrics
from ecg_arrhythmia.preprocessing.beat_extraction import BEAT_SYMBOLS


def test_xqrs_detects_r_peaks_in_real_mit_bih_record():
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

    detector = XQRSDetector(learn=True)

    detected_samples = detector.detect(
        signal=signal,
        sampling_rate=sampling_rate,
    )

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

    assert detected_samples.size > 0
    assert detected_samples.dtype == np.int64
    assert metrics.recall > 0.99
    assert metrics.precision > 0.99
    assert metrics.mean_absolute_offset_ms is not None
    assert metrics.mean_absolute_offset_ms < 10
