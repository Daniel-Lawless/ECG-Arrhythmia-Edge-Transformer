import json
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.evaluation.r_peak_evaluator import (
    evaluate_r_peak_record,
    evaluate_r_peak_records,
)

LOAD_RECORD_TARGET = "ecg_arrhythmia.evaluation.r_peak_evaluator.load_record"
SELECT_CHANNEL_TARGET = (
    "ecg_arrhythmia.evaluation.r_peak_evaluator.select_signal_channel"
)

# One millisecond-to-sample conversion at the sampling rate used below.
SAMPLING_RATE = 360.0
MILLISECONDS_PER_SAMPLE = 1000.0 / SAMPLING_RATE

# Every synthetic record shares this length; all detections lie inside it.
SIGNAL_LENGTH = 1000


class FakeAnnotation:
    """Minimal stand-in for a WFDB annotation object."""

    def __init__(self, sample: list[int], symbol: list[str]) -> None:
        self.sample = np.asarray(sample, dtype=np.int64)
        self.symbol = list(symbol)


class QueueDetector(RPeakDetector):
    """
    Detector that returns a predetermined detection array per call.

    The evaluator processes records in order, so the queued arrays are
    consumed in the same order as ``record_names``. This keeps the tests
    fully deterministic and detector-agnostic.
    """

    def __init__(self, detections_per_record: list[np.ndarray]) -> None:
        self._queue = list(detections_per_record)

    @property
    def name(self) -> str:
        return "dummy"

    def _detect(
        self,
        signal: np.ndarray,
        sampling_rate: float,
    ) -> np.ndarray:
        return self._queue.pop(0)


# Expert heartbeat annotations for the two synthetic records. Record A
# also contains a non-heartbeat "+" annotation, which must be excluded.
RECORD_ANNOTATIONS = {
    "recA": FakeAnnotation(
        sample=[50, 100, 200, 300],
        symbol=["+", "N", "N", "V"],
    ),
    "recB": FakeAnnotation(
        sample=[400, 600],
        symbol=["N", "S"],
    ),
}


def _fake_load_record(record_name: str):
    # signals is unused because select_signal_channel is mocked.
    return None, {"fs": SAMPLING_RATE}, RECORD_ANNOTATIONS[record_name]


def _fake_select_signal_channel(signals, fields):
    return np.zeros(SIGNAL_LENGTH, dtype=np.float64), "MLII"


def _patched_evaluate_records(detector, record_names, tolerance_ms=100.0):
    with (
        patch(LOAD_RECORD_TARGET, side_effect=_fake_load_record),
        patch(SELECT_CHANNEL_TARGET, side_effect=_fake_select_signal_channel),
    ):
        return evaluate_r_peak_records(
            detector=detector,
            record_names=record_names,
            tolerance_ms=tolerance_ms,
            split_name="validation",
        )


def _two_record_evaluation():
    """
    Build the standard two-record evaluation used across the tests.

    Record A matches all three beats with sample offsets [0, 0, +6].
    Record B matches one beat with offset +10 and misses the S beat, and
    contributes no false positives.
    """

    detector = QueueDetector(
        detections_per_record=[
            np.array([100, 200, 306], dtype=np.int64),
            np.array([410], dtype=np.int64),
        ]
    )

    return _patched_evaluate_records(
        detector=detector,
        record_names=["recA", "recB"],
    )


def test_evaluate_records_aggregates_counts():
    evaluation = _two_record_evaluation()
    metrics = evaluation.metrics

    assert metrics.num_records == 2
    assert metrics.num_annotations == 5
    assert metrics.num_detections == 4
    assert metrics.true_positives == 4
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1


def test_evaluate_records_uses_micro_averaged_metrics():
    evaluation = _two_record_evaluation()
    metrics = evaluation.metrics

    # Micro metrics are computed from the summed counts.
    assert metrics.precision == pytest.approx(4 / 4)
    assert metrics.recall == pytest.approx(4 / 5)
    assert metrics.f1 == pytest.approx(2 * 1.0 * 0.8 / (1.0 + 0.8))

    # The record-level F1 values are 1.0 and 0.6667, whose unweighted
    # mean is 0.8333. The micro F1 must not equal that macro average.
    macro_f1 = (1.0 + (2 * 1.0 * 0.5 / 1.5)) / 2
    assert metrics.f1 != pytest.approx(macro_f1)


def test_evaluate_records_timing_uses_all_matched_beats():
    evaluation = _two_record_evaluation()
    metrics = evaluation.metrics

    # Matched sample offsets across both records are [0, 0, 6, 10]. The
    # dataset mean absolute offset is therefore 4.0 samples.
    expected_all_beats = 4.0 * MILLISECONDS_PER_SAMPLE

    # Averaging each record's mean (2.0 and 10.0) would give 6.0 samples;
    # the aggregate must use all matched beats instead.
    mean_of_record_means = 6.0 * MILLISECONDS_PER_SAMPLE

    assert metrics.mean_absolute_offset_ms == pytest.approx(expected_all_beats)
    assert metrics.mean_absolute_offset_ms != pytest.approx(mean_of_record_means)

    assert metrics.mean_offset_ms == pytest.approx(4.0 * MILLISECONDS_PER_SAMPLE)
    assert metrics.median_absolute_offset_ms == pytest.approx(
        3.0 * MILLISECONDS_PER_SAMPLE
    )
    assert metrics.maximum_absolute_offset_ms == pytest.approx(
        10.0 * MILLISECONDS_PER_SAMPLE
    )


def test_evaluate_records_result_excludes_raw_offsets():
    evaluation = _two_record_evaluation()

    serialised = json.dumps(asdict(evaluation))

    # Only compact timing summaries (singular "offset") should survive;
    # the raw per-beat "offsets" arrays must not be serialised.
    assert "offsets" not in serialised


def test_evaluate_records_aggregates_symbol_metrics():
    evaluation = _two_record_evaluation()
    symbol_metrics = evaluation.symbol_metrics

    # "+" is not a heartbeat symbol and must be excluded entirely.
    assert set(symbol_metrics) == {"N", "V", "S"}

    normal = symbol_metrics["N"]
    assert normal.annotations == 3
    assert normal.matched == 3
    assert normal.missed == 0
    assert normal.recall == pytest.approx(1.0)

    ventricular = symbol_metrics["V"]
    assert ventricular.annotations == 1
    assert ventricular.matched == 1
    assert ventricular.missed == 0

    supraventricular = symbol_metrics["S"]
    assert supraventricular.annotations == 1
    assert supraventricular.matched == 0
    assert supraventricular.missed == 1
    assert supraventricular.recall == pytest.approx(0.0)


def test_evaluate_records_is_detector_agnostic():
    # A minimal custom detector is evaluated without the evaluator
    # importing or branching on any specific detector type.
    evaluation = _two_record_evaluation()

    assert evaluation.detector_name == "dummy"
    assert evaluation.split_name == "validation"
    assert len(evaluation.records) == 2


def test_evaluate_records_empty_matches_produce_none_timing():
    # The detector finds no peaks, so there are no matched beats and the
    # dataset timing statistics must be reported as None.
    detector = QueueDetector(detections_per_record=[np.array([], dtype=np.int64)])

    evaluation = _patched_evaluate_records(
        detector=detector,
        record_names=["recA"],
    )
    metrics = evaluation.metrics

    assert metrics.true_positives == 0
    assert metrics.false_negatives == 3
    assert metrics.mean_offset_ms is None
    assert metrics.mean_absolute_offset_ms is None
    assert metrics.median_absolute_offset_ms is None
    assert metrics.standard_deviation_offset_ms is None
    assert metrics.maximum_absolute_offset_ms is None


def test_evaluate_records_rejects_empty_record_list():
    detector = QueueDetector(detections_per_record=[])

    with pytest.raises(ValueError):
        evaluate_r_peak_records(
            detector=detector,
            record_names=[],
            tolerance_ms=100.0,
            split_name="validation",
        )


@pytest.mark.parametrize(
    "tolerance_ms",
    [-1.0, -100.0, np.nan, np.inf],
)
def test_evaluate_record_rejects_invalid_tolerance(tolerance_ms):
    detector = QueueDetector(detections_per_record=[np.array([100], dtype=np.int64)])

    # Tolerance is validated before any record is loaded, so no mocking
    # of the loader is required here.
    with pytest.raises(ValueError):
        evaluate_r_peak_record(
            detector=detector,
            record_name="recA",
            tolerance_ms=tolerance_ms,
        )


def test_evaluate_single_record_returns_expected_symbol_recall():
    detector = QueueDetector(
        detections_per_record=[np.array([100, 200, 306], dtype=np.int64)]
    )

    with (
        patch(LOAD_RECORD_TARGET, side_effect=_fake_load_record),
        patch(SELECT_CHANNEL_TARGET, side_effect=_fake_select_signal_channel),
    ):
        record_evaluation = evaluate_r_peak_record(
            detector=detector,
            record_name="recA",
            tolerance_ms=100.0,
        )

    assert record_evaluation.record_name == "recA"
    assert record_evaluation.lead_name == "MLII"
    assert record_evaluation.metrics.true_positives == 3
    assert record_evaluation.symbol_metrics["V"].recall == pytest.approx(1.0)
