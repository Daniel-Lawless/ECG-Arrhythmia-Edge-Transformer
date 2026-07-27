import json
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from ecg_arrhythmia.data.build_xqrs_centered_dataset import (
    assert_splits_pairwise_disjoint,
    build_record_detected_beats,
    build_xqrs_centered_dataset,
    load_split_record_names,
    save_xqrs_centered_dataset,
)
from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    SAMPLING_RATE,
    WINDOW_SIZE,
)

LOAD_RECORD_TARGET = "ecg_arrhythmia.data.build_xqrs_centered_dataset.load_record"
SELECT_CHANNEL_TARGET = (
    "ecg_arrhythmia.data.build_xqrs_centered_dataset.select_signal_channel"
)

SAMPLING_RATE_HZ = 360.0
SIGNAL_LENGTH = 1900

# The detector output for the synthetic record. Detection 0 (100) is the RR
# seed, detection 4 (950) is a false positive, and detection 9 (1800) has an
# incomplete window and is dropped at the boundary.
DETECTIONS = [100, 300, 500, 700, 950, 1100, 1300, 1500, 1700, 1800]

# Expert heartbeat annotations. "+" is a non-heartbeat and must be excluded.
# The final "N" at 1850 has no matching detection (a false negative).
ANNOTATION_SAMPLES = [50, 305, 505, 690, 1105, 1290, 1505, 1695, 1850]
ANNOTATION_SYMBOLS = ["+", "N", "V", "N", "/", "S", "N", "N", "N"]


class FakeAnnotation:
    def __init__(self, sample: list[int], symbol: list[str]) -> None:
        self.sample = np.asarray(sample, dtype=np.int64)
        self.symbol = list(symbol)


class DummyDetector(RPeakDetector):
    """Detector returning a fixed detection array for the test record."""

    def __init__(self, detections: list[int]) -> None:
        self._detections = np.asarray(detections, dtype=np.int64)

    @property
    def name(self) -> str:
        return "dummy"

    def _detect(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        return self._detections


@contextmanager
def _mocked_record():
    signal = np.arange(SIGNAL_LENGTH, dtype=np.float64)
    annotation = FakeAnnotation(ANNOTATION_SAMPLES, ANNOTATION_SYMBOLS)

    def fake_load_record(record_name):
        return None, {"fs": SAMPLING_RATE_HZ}, annotation

    def fake_select_signal_channel(signals, fields):
        return signal, "MLII"

    with (
        patch(LOAD_RECORD_TARGET, side_effect=fake_load_record),
        patch(SELECT_CHANNEL_TARGET, side_effect=fake_select_signal_channel),
    ):
        yield signal


def _build_record_conversion():
    with _mocked_record():
        return build_record_detected_beats(
            record_name="synthetic",
            detector=DummyDetector(DETECTIONS),
            tolerance_ms=100.0,
            normalise_beats=False,
            excluded_labels={"Q"},
        )


def test_conversion_counts_are_correct():
    conversion = _build_record_conversion()

    assert conversion.num_annotations == 8
    assert conversion.num_detections == 10
    assert conversion.true_positives == 7
    assert conversion.false_positives == 3
    assert conversion.false_negatives == 1
    assert conversion.matched_classifiable == 6
    assert conversion.unsupported_removed == 1
    assert conversion.boundary_removed == 1
    assert conversion.insufficient_rr_removed == 1
    assert conversion.num_valid_beats == 8


def test_labels_are_transferred_from_matched_annotations():
    conversion = _build_record_conversion()

    # Detection 0 (seed) and 9 (boundary) are dropped; the eight valid beats
    # correspond to detections at 300, 500, 700, 950, 1100, 1300, 1500, 1700.
    np.testing.assert_array_equal(
        conversion.is_matched,
        np.array([True, True, True, False, True, True, True, True]),
    )
    np.testing.assert_array_equal(
        conversion.is_target,
        np.array([True, True, True, False, False, True, True, True]),
    )

    # The false-positive detection (950) receives no label.
    assert conversion.symbols[3] == ""
    assert conversion.aami_labels[3] == ""

    # The matched "/" beat maps to Q and is excluded from scored targets
    # but remains a matched context beat.
    assert conversion.symbols[4] == "/"
    assert conversion.aami_labels[4] == "Q"
    assert not bool(conversion.is_target[4])

    # Matched, classifiable beats keep their transferred expert class.
    assert list(conversion.aami_labels[[0, 1, 2, 5, 6, 7]]) == [
        "N",
        "V",
        "N",
        "S",
        "N",
        "N",
    ]


def test_windows_are_centred_on_detected_positions():
    conversion = _build_record_conversion()

    assert conversion.windows.shape == (8, WINDOW_SIZE)

    valid_detections = [300, 500, 700, 950, 1100, 1300, 1500, 1700]
    for beat_index, detected_sample in enumerate(valid_detections):
        expected_window = np.arange(
            detected_sample - SAMPLES_BEFORE,
            detected_sample + SAMPLES_AFTER,
            dtype=np.float64,
        )
        np.testing.assert_array_equal(conversion.windows[beat_index], expected_window)


def test_offsets_use_detected_minus_annotation():
    conversion = _build_record_conversion()

    # detected - expert annotation, in samples, for each valid beat.
    np.testing.assert_array_equal(
        conversion.offset_samples,
        np.array([-5, -5, 10, 0, -5, 10, -5, 5]),
    )

    # The transferred label is unaffected by the timing offset: beat 2 is a
    # late detection (+10) but still carries the expert "N" class.
    assert conversion.aami_labels[2] == "N"


def test_rr_features_use_the_full_detection_timeline():
    conversion = _build_record_conversion()

    # Beat index 4 is detection 1100. Its previous detection is the false
    # positive at 950, so the previous RR interval is (1100 - 950) samples.
    # If false positives were excluded, the previous detection would be 700
    # and the interval would be (1100 - 700) samples instead.
    with_false_positive = (1100 - 950) / SAMPLING_RATE
    without_false_positive = (1100 - 700) / SAMPLING_RATE

    assert conversion.rr_features[4, 0] == pytest.approx(with_false_positive)
    assert conversion.rr_features[4, 0] != pytest.approx(without_false_positive)

    # RR features are always positive after clipping.
    assert np.all(conversion.rr_features > 0)


def test_build_dataset_scores_only_matched_classifiable_targets():
    with _mocked_record():
        dataset = build_xqrs_centered_dataset(
            record_names=["synthetic"],
            detector=DummyDetector(DETECTIONS),
            tolerance_ms=100.0,
        )

    # Eight valid beats give four length-5 windows; only three have a
    # matched, classifiable target beat (S, N, N).
    assert dataset.X_sequences.shape == (3, 5, WINDOW_SIZE)
    assert dataset.rr_sequences.shape == (3, 5, 2)
    assert list(dataset.y_labels) == ["S", "N", "N"]

    # Audit arrays align with the kept sequences.
    np.testing.assert_array_equal(
        dataset.audit_offset_samples,
        np.array([10, -5, 5]),
    )

    # Every kept sequence includes the false-positive detection in context.
    np.testing.assert_array_equal(
        dataset.audit_has_unmatched_context,
        np.array([True, True, True]),
    )


def test_build_dataset_summary_has_expected_statistics():
    with _mocked_record():
        dataset = build_xqrs_centered_dataset(
            record_names=["synthetic"],
            detector=DummyDetector(DETECTIONS),
            tolerance_ms=100.0,
        )

    summary = dataset.summary
    assert summary["total_annotations"] == 8
    assert summary["total_detections"] == 10
    assert summary["true_positives"] == 7
    assert summary["false_positives"] == 3
    assert summary["false_negatives"] == 1
    assert summary["matched_classifiable_detections"] == 6
    assert summary["unsupported_labels_removed"] == 1
    assert summary["boundary_windows_removed"] == 1
    assert summary["insufficient_rr_history_removed"] == 1
    assert summary["num_valid_detected_beats"] == 8
    assert summary["num_final_sequences"] == 3
    assert summary["num_sequences_with_unmatched_context"] == 3
    assert summary["target_class_distribution"] == {"N": 2, "S": 1}


def test_saved_dataset_does_not_serialise_raw_offsets(tmp_path):
    with _mocked_record():
        dataset = build_xqrs_centered_dataset(
            record_names=["synthetic"],
            detector=DummyDetector(DETECTIONS),
            tolerance_ms=100.0,
        )

    output_dir = tmp_path / "xqrs_val"
    save_xqrs_centered_dataset(dataset=dataset, output_dir=output_dir)

    # Core sequence arrays and compact audit arrays are written.
    assert (output_dir / "X.npy").exists()
    assert (output_dir / "y.npy").exists()
    assert (output_dir / "rr_features.npy").exists()
    assert (output_dir / "audit_offset_ms.npy").exists()

    # The raw per-beat offsets live in .npy files, never in the JSON summary.
    summary_text = (output_dir / "dataset_summary.json").read_text(encoding="utf-8")
    assert "offset" not in summary_text
    parsed = json.loads(summary_text)
    assert "target_class_distribution" in parsed


def test_sampling_rate_constant_matches_the_expert_pipeline():
    # The reused beat extractor computes RR using the MIT-BIH sampling rate,
    # so the synthetic record must use the same rate for a fair comparison.
    assert SAMPLING_RATE == 360


# ---------------------------------------------------------------------
#                    Split-Name Record Loading
# ---------------------------------------------------------------------

SPLIT_SUMMARY = {
    "per_split": {
        "train": {"selected_patient_ids": ["101", "201_202", "203"]},
        "val": {"selected_patient_ids": ["114", "122"]},
        "test": {"selected_patient_ids": ["100", "207_223"]},
    }
}


def _write_split_summary(tmp_path):
    summary_path = tmp_path / "split_summary_metrics.json"
    summary_path.write_text(json.dumps(SPLIT_SUMMARY), encoding="utf-8")
    return summary_path


def test_load_train_records_expand_grouped_ids(tmp_path):
    summary_path = _write_split_summary(tmp_path)

    records = load_split_record_names(summary_path, "train")

    # "201_202" expands into the underlying raw records, order preserved.
    assert records == ["101", "201", "202", "203"]


def test_load_val_records(tmp_path):
    summary_path = _write_split_summary(tmp_path)

    assert load_split_record_names(summary_path, "val") == ["114", "122"]


def test_train_and_val_records_are_disjoint(tmp_path):
    summary_path = _write_split_summary(tmp_path)

    train_records = set(load_split_record_names(summary_path, "train"))
    val_records = set(load_split_record_names(summary_path, "val"))

    assert train_records.isdisjoint(val_records)


def test_load_test_records_expand_grouped_ids(tmp_path):
    summary_path = _write_split_summary(tmp_path)

    # The test split is now supported and grouped IDs expand.
    assert load_split_record_names(summary_path, "test") == ["100", "207", "223"]


def test_all_splits_are_pairwise_disjoint(tmp_path):
    summary_path = _write_split_summary(tmp_path)

    split_records = assert_splits_pairwise_disjoint(summary_path)

    assert set(split_records["train"]).isdisjoint(split_records["val"])
    assert set(split_records["train"]).isdisjoint(split_records["test"])
    assert set(split_records["val"]).isdisjoint(split_records["test"])


def test_pairwise_disjoint_raises_on_overlap(tmp_path):
    overlapping = {
        "per_split": {
            "train": {"selected_patient_ids": ["101", "100"]},
            "val": {"selected_patient_ids": ["114"]},
            "test": {"selected_patient_ids": ["100"]},
        }
    }
    summary_path = tmp_path / "overlap.json"
    summary_path.write_text(json.dumps(overlapping), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        assert_splits_pairwise_disjoint(summary_path)


def test_load_split_records_rejects_unknown_split(tmp_path):
    summary_path = _write_split_summary(tmp_path)

    with pytest.raises(ValueError, match="Unsupported split name"):
        load_split_record_names(summary_path, "holdout")


def test_split_name_is_written_into_summary():
    with _mocked_record():
        dataset = build_xqrs_centered_dataset(
            record_names=["synthetic"],
            detector=DummyDetector(DETECTIONS),
            tolerance_ms=100.0,
            split_name="train",
        )

    assert dataset.summary["split_name"] == "train"


def test_test_split_name_is_written_into_summary():
    with _mocked_record():
        dataset = build_xqrs_centered_dataset(
            record_names=["synthetic"],
            detector=DummyDetector(DETECTIONS),
            tolerance_ms=100.0,
            split_name="test",
        )

    assert dataset.summary["split_name"] == "test"
