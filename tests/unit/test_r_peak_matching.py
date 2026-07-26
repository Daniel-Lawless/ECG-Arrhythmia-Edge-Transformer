import numpy as np

from ecg_arrhythmia.evaluation.r_peak_matching import match_r_peaks


def test_match_r_peaks_matches_exact_peak_positions():
    annotation_samples = np.array(
        [100, 300, 500],
        dtype=np.int64,
    )
    detected_samples = np.array(
        [100, 300, 500],
        dtype=np.int64,
    )

    result = match_r_peaks(
        annotation_samples=annotation_samples,
        detected_samples=detected_samples,
        tolerance_samples=10,
    )

    np.testing.assert_array_equal(
        result.matched_annotation_indices,
        np.array([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        result.matched_detection_indices,
        np.array([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        result.offsets_samples,
        np.array([0, 0, 0]),
    )

    assert result.true_positives == 3
    assert result.false_negatives == 0
    assert result.false_positives == 0


def test_match_r_peaks_records_early_and_late_offsets():
    result = match_r_peaks(
        annotation_samples=np.array([100, 300]),
        detected_samples=np.array([95, 307]),
        tolerance_samples=10,
    )

    # Detection 95 is five samples early.
    # Detection 307 is seven samples late.
    np.testing.assert_array_equal(
        result.offsets_samples,
        np.array([-5, 7]),
    )


def test_match_r_peaks_identifies_false_negatives_and_false_positives():
    result = match_r_peaks(
        annotation_samples=np.array([100, 300, 500]),
        detected_samples=np.array([104, 480, 700]),
        tolerance_samples=25,
    )

    np.testing.assert_array_equal(
        result.matched_annotation_indices,
        np.array([0, 2]),
    )
    np.testing.assert_array_equal(
        result.matched_detection_indices,
        np.array([0, 1]),
    )
    np.testing.assert_array_equal(
        result.unmatched_annotation_indices,
        np.array([1]),
    )
    np.testing.assert_array_equal(
        result.unmatched_detection_indices,
        np.array([2]),
    )

    assert result.true_positives == 2
    assert result.false_negatives == 1
    assert result.false_positives == 1


def test_match_r_peaks_does_not_reuse_a_detection():
    result = match_r_peaks(
        annotation_samples=np.array([100, 110]),
        detected_samples=np.array([105]),
        tolerance_samples=10,
    )

    # The single detection can only be assigned once.
    assert result.true_positives == 1
    assert result.false_negatives == 1
    assert result.false_positives == 0


def test_match_r_peaks_preserves_chronological_matching():
    result = match_r_peaks(
        annotation_samples=np.array([100, 110]),
        detected_samples=np.array([94, 104]),
        tolerance_samples=6,
    )

    # Chronological matching produces two valid one-to-one pairs:
    # 100 ↔ 94 and 110 ↔ 104.
    assert result.true_positives == 2

    np.testing.assert_array_equal(
        result.offsets_samples,
        np.array([-6, -6]),
    )


def test_match_r_peaks_accepts_empty_detection_array():
    result = match_r_peaks(
        annotation_samples=np.array([100, 300]),
        detected_samples=np.array([], dtype=np.int64),
        tolerance_samples=10,
    )

    assert result.true_positives == 0
    assert result.false_negatives == 2
    assert result.false_positives == 0
