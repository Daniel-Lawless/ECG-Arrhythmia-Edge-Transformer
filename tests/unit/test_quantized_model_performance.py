import numpy as np
import pytest

from ecg_arrhythmia.evaluation.evaluate_quantized_model_performance import (
    align_ground_truth,
    build_aggregate,
    changed_outcomes,
    classification_metrics,
    margin_medians,
    metric_deltas,
    transition_outcomes,
)
from ecg_arrhythmia.evaluation.quantized_model_performance_plots import (
    write_performance_figures,
)

# Class order: N=0, S=1, V=2, F=3.
TRUE = np.array([0, 0, 1, 2, 3, 0], dtype=np.int64)
FP32 = np.array([0, 0, 1, 2, 0, 1], dtype=np.int64)
INT8 = np.array([0, 1, 1, 2, 3, 1], dtype=np.int64)
# Changed at position 1 (FP32 correct N -> INT8 wrong S) and position 4
# (FP32 wrong N -> INT8 correct F). Position 5 is wrong under both but
# unchanged, so it is not a changed prediction.


# ---------------------------------------------------------------------
#                        Ground-Truth Alignment
# ---------------------------------------------------------------------


def test_labels_resolve_by_target_peak_and_unmatched_are_excluded():
    labels_by_peak = {1000: "N", 3000: "V"}

    positions, true_indices = align_ground_truth(
        [1000, 2000, 3000],
        labels_by_peak,
    )

    np.testing.assert_array_equal(positions, [0, 2])
    np.testing.assert_array_equal(true_indices, [0, 2])


def test_an_unsupported_ground_truth_label_is_rejected():
    with pytest.raises(ValueError, match="Unsupported ground-truth label"):
        align_ground_truth([1000], {1000: "Q"})


def test_the_same_positions_index_both_models():
    # The labelled positions are one array applied to both prediction
    # arrays, so FP32 and INT8 are scored on exactly the same subset.
    positions, _ = align_ground_truth([10, 20, 30], {20: "S", 30: "F"})

    fp32 = np.array([0, 1, 3])[positions]
    int8 = np.array([2, 1, 0])[positions]

    np.testing.assert_array_equal(fp32, [1, 3])
    np.testing.assert_array_equal(int8, [1, 0])


# ---------------------------------------------------------------------
#                        Classification Metrics
# ---------------------------------------------------------------------


def test_metrics_match_hand_computed_values():
    metrics = classification_metrics(
        np.array([0, 0, 1, 2]),
        np.array([0, 1, 1, 2]),
    )

    assert metrics["num_sequences"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)

    n = metrics["per_class"]["N"]
    assert n["precision"] == pytest.approx(1.0)
    assert n["recall"] == pytest.approx(0.5)
    assert n["f1"] == pytest.approx(2 / 3)
    assert n["support"] == 2

    s = metrics["per_class"]["S"]
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(1.0)

    # F has no support: zero_division=0 keeps its metrics at zero.
    assert metrics["per_class"]["F"]["support"] == 0
    assert metrics["per_class"]["F"]["f1"] == 0.0

    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1.0 + 0.0) / 4)

    # Rows are ground truth, columns are predictions.
    assert metrics["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]


def test_metrics_reject_mismatched_or_empty_inputs():
    with pytest.raises(ValueError, match="matching 1-D arrays"):
        classification_metrics(np.array([0, 1]), np.array([0]))

    with pytest.raises(ValueError, match="At least one labelled sequence"):
        classification_metrics(np.array([], dtype=np.int64), np.array([]))


def test_deltas_are_signed_int8_minus_fp32():
    fp32_metrics = classification_metrics(TRUE, FP32)
    int8_metrics = classification_metrics(TRUE, INT8)

    deltas = metric_deltas(fp32_metrics, int8_metrics)

    assert deltas["accuracy_delta"] == pytest.approx(
        int8_metrics["accuracy"] - fp32_metrics["accuracy"]
    )
    assert deltas["macro_f1_delta"] == pytest.approx(
        int8_metrics["macro_f1"] - fp32_metrics["macro_f1"]
    )
    assert deltas["per_class_deltas"]["F"]["recall_delta"] == pytest.approx(
        int8_metrics["per_class"]["F"]["recall"]
        - fp32_metrics["per_class"]["F"]["recall"]
    )

    expected_delta = np.asarray(int8_metrics["confusion_matrix"]) - np.asarray(
        fp32_metrics["confusion_matrix"]
    )
    assert deltas["confusion_matrix_delta"] == expected_delta.tolist()


# ---------------------------------------------------------------------
#                     Changed-Prediction Outcomes
# ---------------------------------------------------------------------


def test_changed_predictions_are_categorised_exactly():
    outcomes = changed_outcomes(TRUE, FP32, INT8)

    assert outcomes["num_changed"] == 2
    assert outcomes["fp32_correct_int8_wrong"] == 1
    assert outcomes["fp32_wrong_int8_correct"] == 1
    assert outcomes["both_wrong"] == 0
    assert outcomes["net_correct_change"] == 0

    by_class = outcomes["by_ground_truth_class"]
    assert by_class["N"]["fp32_correct_int8_wrong"] == 1
    assert by_class["F"]["fp32_wrong_int8_correct"] == 1
    assert by_class["S"] == {
        "fp32_correct_int8_wrong": 0,
        "fp32_wrong_int8_correct": 0,
        "both_wrong": 0,
    }


def test_a_both_wrong_change_is_neither_gain_nor_loss():
    true = np.array([2, 2])
    fp32 = np.array([0, 2])
    int8 = np.array([1, 2])

    outcomes = changed_outcomes(true, fp32, int8)

    assert outcomes["num_changed"] == 1
    assert outcomes["both_wrong"] == 1
    assert outcomes["net_correct_change"] == 0


def test_net_change_equals_the_difference_in_correct_predictions():
    outcomes = changed_outcomes(TRUE, FP32, INT8)

    fp32_correct = int(np.sum(FP32 == TRUE))
    int8_correct = int(np.sum(INT8 == TRUE))

    assert outcomes["net_correct_change"] == int8_correct - fp32_correct


def test_transitions_are_annotated_with_ground_truth():
    outcomes = transition_outcomes(TRUE, FP32, INT8)

    # Position 1: FP32 N -> INT8 S with truth N.
    assert outcomes["N_to_S"] == {
        "count": 1,
        "fp32_correct_int8_wrong": 1,
        "fp32_wrong_int8_correct": 0,
        "both_wrong": 0,
    }
    # Position 4: FP32 N -> INT8 F with truth F.
    assert outcomes["N_to_F"] == {
        "count": 1,
        "fp32_correct_int8_wrong": 0,
        "fp32_wrong_int8_correct": 1,
        "both_wrong": 0,
    }
    # Only observed transitions are reported.
    assert "S_to_N" not in outcomes


def test_a_transition_where_both_are_wrong_is_counted_as_such():
    outcomes = transition_outcomes(
        np.array([2]),
        np.array([0]),
        np.array([1]),
    )

    assert outcomes["N_to_S"] == {
        "count": 1,
        "fp32_correct_int8_wrong": 0,
        "fp32_wrong_int8_correct": 0,
        "both_wrong": 1,
    }


def test_margin_medians_are_split_by_outcome_category():
    margins = np.array([5.0, 0.5, 4.0, 3.0, 0.2, 1.0], dtype=np.float32)

    medians = margin_medians(margins, TRUE, FP32, INT8)

    # Agreeing positions are 0, 2, 3 and 5 with margins 5, 4, 3 and 1.
    assert medians["agreeing"] == pytest.approx(3.5)
    assert medians["fp32_correct_int8_wrong"] == pytest.approx(0.5)
    assert medians["fp32_wrong_int8_correct"] == pytest.approx(0.2)
    assert medians["both_wrong"] is None


# ---------------------------------------------------------------------
#                        Aggregate Pooling
# ---------------------------------------------------------------------


def _record_result(name, observed, labelled, changed_excluded=0):
    return {
        "record_name": name,
        "streaming_sequences_observed": observed,
        "labelled_sequences_evaluated": labelled,
        "unlabelled_sequences_excluded": observed - labelled,
        "changed_predictions_excluded_from_ground_truth": changed_excluded,
    }


def test_aggregate_metrics_come_from_pooled_arrays():
    # Record A: one sequence, correct. Record B: three sequences, one
    # correct. Averaging record accuracies would give 0.6667; pooling
    # gives 0.5.
    pooled = {
        "true": np.array([0, 0, 0, 0], dtype=np.int64),
        "fp32": np.array([0, 1, 1, 0], dtype=np.int64),
        "int8": np.array([0, 1, 1, 1], dtype=np.int64),
        "fp32_margins": np.array([3.0, 1.0, 1.0, 0.5], dtype=np.float32),
    }

    aggregate = build_aggregate(
        record_results=[
            _record_result("114", 2, 1),
            _record_result("122", 3, 3, changed_excluded=1),
        ],
        pooled=pooled,
    )

    assert aggregate["total_streaming_sequences_observed"] == 5
    assert aggregate["total_labelled_sequences_evaluated"] == 4
    assert aggregate["total_unlabelled_sequences_excluded"] == 1
    assert aggregate["changed_predictions_excluded_from_ground_truth"] == 1

    assert aggregate["fp32"]["accuracy"] == pytest.approx(0.5)
    assert aggregate["int8"]["accuracy"] == pytest.approx(0.25)
    assert aggregate["deltas"]["accuracy_delta"] == pytest.approx(-0.25)
    assert aggregate["changed_outcomes"]["fp32_correct_int8_wrong"] == 1
    assert aggregate["changed_outcomes"]["net_correct_change"] == -1


# ---------------------------------------------------------------------
#                                Plots
# ---------------------------------------------------------------------


def test_performance_figures_render_for_a_tiny_example(tmp_path):
    fp32_metrics = classification_metrics(TRUE, FP32)
    int8_metrics = classification_metrics(TRUE, INT8)

    written = write_performance_figures(
        fp32_metrics=fp32_metrics,
        int8_metrics=int8_metrics,
        deltas=metric_deltas(fp32_metrics, int8_metrics),
        outcomes=changed_outcomes(TRUE, FP32, INT8),
        figures_dir=tmp_path,
    )

    # Two confusion matrices, F1 and recall comparisons, deltas, outcomes.
    assert len(written) == 6
    assert all(path.exists() for path in written)
    assert {path.suffix for path in tmp_path.iterdir()} == {".png"}
