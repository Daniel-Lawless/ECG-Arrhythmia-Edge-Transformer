import numpy as np
import pytest

from ecg_arrhythmia.evaluation.paired_centering_comparison import (
    compare_paired_metrics,
    compute_correctness_transitions,
    compute_prediction_agreement,
)

# Shared synthetic paired predictions (class indices N=0, S=1, V=2, F=3).
TRUE = np.array([0, 1, 2, 0, 1], dtype=np.int64)
EXPERT = np.array([0, 1, 2, 0, 2], dtype=np.int64)
XQRS = np.array([0, 2, 2, 1, 1], dtype=np.int64)


def _metrics(loss, accuracy, macro_f1, per_class_f1):
    """Build a minimal metrics dict for compare_paired_metrics."""

    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {label: {"f1": f1} for label, f1 in per_class_f1.items()},
    }


def test_compare_paired_metrics_reports_absolute_changes_only():
    expert = _metrics(0.30, 0.97, 0.69, {"N": 0.98, "S": 0.83, "V": 0.94, "F": 0.01})
    xqrs = _metrics(0.33, 0.96, 0.68, {"N": 0.98, "S": 0.79, "V": 0.93, "F": 0.00})

    change = compare_paired_metrics(expert, xqrs, sequence_count=14548)

    # Only the two change blocks are present (no relative change, no counts).
    assert set(change) == {"overall_change", "per_class_f1_change"}

    assert change["overall_change"]["loss"] == pytest.approx(0.03)
    assert change["overall_change"]["accuracy"] == pytest.approx(-0.01)
    assert change["overall_change"]["macro_f1"] == pytest.approx(-0.01)

    assert set(change["per_class_f1_change"]) == {"N", "S", "V", "F"}
    assert change["per_class_f1_change"]["S"] == pytest.approx(-0.04)
    assert change["per_class_f1_change"]["F"] == pytest.approx(-0.01)


def test_prediction_agreement():
    agreement = compute_prediction_agreement(EXPERT, XQRS)

    assert set(agreement) == {
        "num_targets",
        "identical_count",
        "identical_fraction",
        "changed_count",
    }
    assert agreement["num_targets"] == 5
    assert agreement["identical_count"] == 2
    assert agreement["changed_count"] == 3
    assert agreement["identical_fraction"] == pytest.approx(0.4)


def test_correctness_transitions():
    transitions = compute_correctness_transitions(TRUE, EXPERT, XQRS)

    assert set(transitions) == {
        "correct_to_correct",
        "correct_to_incorrect",
        "incorrect_to_correct",
        "incorrect_to_incorrect",
    }
    assert transitions["correct_to_correct"] == 2
    assert transitions["correct_to_incorrect"] == 2
    assert transitions["incorrect_to_correct"] == 1
    assert transitions["incorrect_to_incorrect"] == 0

    # The four counts must partition every paired target.
    assert sum(transitions.values()) == TRUE.size
