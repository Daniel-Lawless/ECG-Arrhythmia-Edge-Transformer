import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.training.transformer_training import CLASS_LABELS

# ---------------------------------------------------------------------
#                        Compact Metric Comparison
# ---------------------------------------------------------------------


def _delta(expert_value: float, xqrs_value: float) -> float:
    """Return the absolute change from the expert to the XQRS condition."""

    return round(xqrs_value - expert_value, 4)


def compare_paired_metrics(
    expert_metrics: dict,
    xqrs_metrics: dict,
    sequence_count: int,
) -> dict[str, dict[str, float]]:
    """
    Build the compact paired metric comparison.

    Because the paired sequence count is already proven equal, only the
    absolute metric changes are reported; no sequence-count delta is
    included here.
    """

    return {
        "overall_change": {
            "loss": _delta(expert_metrics["loss"], xqrs_metrics["loss"]),
            "accuracy": _delta(expert_metrics["accuracy"], xqrs_metrics["accuracy"]),
            "macro_f1": _delta(expert_metrics["macro_f1"], xqrs_metrics["macro_f1"]),
        },
        "per_class_f1_change": {
            label: _delta(
                expert_metrics["per_class"][label]["f1"],
                xqrs_metrics["per_class"][label]["f1"],
            )
            for label in CLASS_LABELS
        },
    }


# ---------------------------------------------------------------------
#                        Prediction Agreement
# ---------------------------------------------------------------------


def compute_prediction_agreement(
    expert_predictions: NDArray[np.int64],
    xqrs_predictions: NDArray[np.int64],
) -> dict[str, object]:
    """Report how often the two conditions predict the same class."""

    total = int(expert_predictions.size)
    identical = int(np.sum(expert_predictions == xqrs_predictions))
    changed = total - identical

    return {
        "num_targets": total,
        "identical_count": identical,
        "identical_fraction": round(identical / total, 4) if total else None,
        "changed_count": changed,
    }


# ---------------------------------------------------------------------
#                        Correctness Transitions
# ---------------------------------------------------------------------


def compute_correctness_transitions(
    true_indices: NDArray[np.int64],
    expert_predictions: NDArray[np.int64],
    xqrs_predictions: NDArray[np.int64],
) -> dict[str, int]:
    """
    Count how correctness changes from the expert to the XQRS condition.

    "expert" refers to condition A and "xqrs" to condition B.
    """

    expert_correct = expert_predictions == true_indices
    xqrs_correct = xqrs_predictions == true_indices

    return {
        "correct_to_correct": int(np.sum(expert_correct & xqrs_correct)),
        "correct_to_incorrect": int(np.sum(expert_correct & ~xqrs_correct)),
        "incorrect_to_correct": int(np.sum(~expert_correct & xqrs_correct)),
        "incorrect_to_incorrect": int(np.sum(~expert_correct & ~xqrs_correct)),
    }
