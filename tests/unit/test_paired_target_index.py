import numpy as np
import pytest

from ecg_arrhythmia.evaluation.paired_target_index import (
    build_paired_target_index,
    build_pairing_summary,
    create_paired_dataset_views,
)

# Expert targets, in one order.
EXPERT_RECORDS = np.array(["114", "114", "122", "122"])
EXPERT_SAMPLES = np.array([100, 300, 500, 700], dtype=np.int64)
EXPERT_LABELS = np.array(["N", "V", "S", "N"])

# XQRS targets, deliberately in a different order, missing (122, 700) and
# adding (209, 900).
XQRS_RECORDS = np.array(["114", "114", "122", "209"])
XQRS_SAMPLES = np.array([300, 100, 500, 900], dtype=np.int64)
XQRS_LABELS = np.array(["V", "N", "S", "N"])


def _paired_index():
    return build_paired_target_index(
        expert_records=EXPERT_RECORDS,
        expert_annotation_samples=EXPERT_SAMPLES,
        xqrs_records=XQRS_RECORDS,
        xqrs_annotation_samples=XQRS_SAMPLES,
    )


def test_pairing_uses_identity_not_position():
    paired = _paired_index()

    # The shared targets are the intersection of identities, sorted.
    assert paired.paired_keys == [("114", 100), ("114", 300), ("122", 500)]

    # Expert rows are in natural order, but XQRS rows are reordered to the
    # matching identities, proving pairing does not rely on position.
    np.testing.assert_array_equal(paired.expert_rows, np.array([0, 1, 2]))
    np.testing.assert_array_equal(paired.xqrs_rows, np.array([1, 0, 2]))


def test_pairing_reports_exclusions_from_each_side():
    paired = _paired_index()

    assert paired.expert_only_keys == [("122", 700)]
    assert paired.xqrs_only_keys == [("209", 900)]
    assert paired.num_paired == 3
    assert paired.num_expert_targets == 4
    assert paired.num_xqrs_targets == 4


def test_pairing_is_order_independent():
    # Shuffle both sides; the paired keys and their identity mapping must
    # be unchanged.
    expert_perm = [3, 1, 0, 2]
    xqrs_perm = [2, 0, 3, 1]

    paired = build_paired_target_index(
        expert_records=EXPERT_RECORDS[expert_perm],
        expert_annotation_samples=EXPERT_SAMPLES[expert_perm],
        xqrs_records=XQRS_RECORDS[xqrs_perm],
        xqrs_annotation_samples=XQRS_SAMPLES[xqrs_perm],
    )

    assert paired.paired_keys == [("114", 100), ("114", 300), ("122", 500)]

    # Regardless of ordering, the paired rows must recover the same
    # identities on both sides.
    expert_keys = [
        (EXPERT_RECORDS[expert_perm][row], int(EXPERT_SAMPLES[expert_perm][row]))
        for row in paired.expert_rows
    ]
    assert expert_keys == [("114", 100), ("114", 300), ("122", 500)]


def test_duplicate_identities_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        build_paired_target_index(
            expert_records=np.array(["114", "114"]),
            expert_annotation_samples=np.array([100, 100], dtype=np.int64),
            xqrs_records=np.array(["114"]),
            xqrs_annotation_samples=np.array([100], dtype=np.int64),
        )


def test_create_views_aligns_inputs_and_agrees_on_labels():
    paired = _paired_index()

    expert_X = np.array([0.0, 1.0, 2.0, 3.0])
    xqrs_X = np.array([0.0, 10.0, 20.0, 30.0])
    dummy_rr = np.zeros(4)
    offsets = np.array([0, 5, -3, 9], dtype=np.int64)

    views = create_paired_dataset_views(
        paired_index=paired,
        expert_X=expert_X,
        expert_rr=dummy_rr,
        expert_y=EXPERT_LABELS,
        xqrs_X=xqrs_X,
        xqrs_rr=dummy_rr,
        xqrs_y=XQRS_LABELS,
        xqrs_records=XQRS_RECORDS,
        xqrs_annotation_samples=XQRS_SAMPLES,
        xqrs_offset_samples=offsets,
        xqrs_offset_ms=offsets.astype(np.float64),
        xqrs_has_unmatched_context=np.array([False, True, False, True]),
    )

    # Labels agree for every paired target.
    np.testing.assert_array_equal(views.expert_y, np.array(["N", "V", "S"]))
    np.testing.assert_array_equal(views.xqrs_y, np.array(["N", "V", "S"]))

    # Expert rows [0, 1, 2] and XQRS rows [1, 0, 2] select the right inputs.
    np.testing.assert_array_equal(views.expert_X, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(views.xqrs_X, np.array([10.0, 0.0, 20.0]))

    # Audit is taken from the XQRS rows in paired order.
    np.testing.assert_array_equal(views.offset_samples, np.array([5, 0, -3]))
    np.testing.assert_array_equal(
        views.has_unmatched_context, np.array([True, False, False])
    )


def test_create_views_rejects_label_disagreement():
    paired = _paired_index()

    # Corrupt the XQRS label for identity (114, 100) so it disagrees.
    corrupted_labels = np.array(["V", "S", "S", "N"])
    dummy = np.zeros(4)

    with pytest.raises(ValueError, match="labels disagree"):
        create_paired_dataset_views(
            paired_index=paired,
            expert_X=dummy,
            expert_rr=dummy,
            expert_y=EXPERT_LABELS,
            xqrs_X=dummy,
            xqrs_rr=dummy,
            xqrs_y=corrupted_labels,
            xqrs_records=XQRS_RECORDS,
            xqrs_annotation_samples=XQRS_SAMPLES,
            xqrs_offset_samples=np.zeros(4, dtype=np.int64),
            xqrs_offset_ms=np.zeros(4),
            xqrs_has_unmatched_context=np.zeros(4, dtype=bool),
        )


def test_pairing_summary_reports_counts_and_exclusions():
    paired = _paired_index()
    paired_labels = np.array(["N", "V", "S"])

    summary = build_pairing_summary(paired_index=paired, paired_labels=paired_labels)

    # The summary is compact: only counts and paired class support.
    assert set(summary) == {
        "total_expert_targets",
        "total_xqrs_targets",
        "paired_targets",
        "expert_only_excluded",
        "xqrs_only_excluded",
        "paired_class_support",
    }
    assert summary["total_expert_targets"] == 4
    assert summary["total_xqrs_targets"] == 4
    assert summary["paired_targets"] == 3
    assert summary["expert_only_excluded"] == 1
    assert summary["xqrs_only_excluded"] == 1
    assert summary["paired_class_support"] == {"N": 1, "S": 1, "V": 1}
