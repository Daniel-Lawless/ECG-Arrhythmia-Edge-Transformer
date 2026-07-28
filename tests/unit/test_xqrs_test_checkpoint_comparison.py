import hashlib
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest
import torch

from ecg_arrhythmia.evaluation.evaluate_xqrs_test_checkpoints import (
    build_test_comparison,
    compute_test_changes,
    load_checkpoint_model,
    sha256_file,
)
from ecg_arrhythmia.evaluation.paired_centering_comparison import (
    compute_correctness_transitions,
    compute_prediction_agreement,
)
from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer
from ecg_arrhythmia.training.transformer_training import (
    ClassMetrics,
    EvaluationMetrics,
)


class OverallChange(TypedDict):
    loss: float
    accuracy: float
    macro_f1: float


class PerClassChange(TypedDict):
    precision: float
    recall: float
    f1: float
    support: int


class ChangesResult(TypedDict):
    overall: OverallChange
    per_class: dict[str, PerClassChange]


class CheckpointResult(TypedDict):
    checkpoint_path: str
    checkpoint_sha256: str
    metrics: dict[str, object]


class ComparisonResult(TypedDict):
    evaluation_type: str
    test_dataset_dir: str
    test_dataset_summary: dict[str, object]
    configuration: dict[str, object]
    original: CheckpointResult
    expc: CheckpointResult
    change: dict[str, object]
    prediction_agreement: dict[str, object]
    correctness_transitions: dict[str, int]


def _per_class(
    precision: float,
    recall: float,
    f1: float,
    support: int,
) -> ClassMetrics:
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_class_count": support,
    }


def _metrics(
    loss: float,
    accuracy: float,
    macro_f1: float,
    per_class: dict[str, ClassMetrics],
) -> EvaluationMetrics:
    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": [[0, 0, 0, 0] for _ in range(4)],
    }


# ---------------------------------------------------------------------
#                       Checkpoint Loading
# ---------------------------------------------------------------------


def test_load_checkpoint_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint_model(
            tmp_path / "missing.pt",
            num_layers=1,
            dropout=0.3,
            device=torch.device("cpu"),
        )


def test_load_checkpoint_loads_matching_architecture(tmp_path: Path):
    model = ECGSequenceTransformer(
        num_classes=4,
        num_layers=1,
        dropout=0.3,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = load_checkpoint_model(
        checkpoint_path,
        num_layers=1,
        dropout=0.3,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, ECGSequenceTransformer)


def test_load_checkpoint_is_strict(tmp_path: Path):
    # A checkpoint from a different depth must not load into the skeleton.
    other = ECGSequenceTransformer(
        num_classes=4,
        num_layers=2,
        dropout=0.3,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(other.state_dict(), checkpoint_path)

    with pytest.raises(RuntimeError):
        load_checkpoint_model(
            checkpoint_path,
            num_layers=1,
            dropout=0.3,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------
#                         Metric Changes
# ---------------------------------------------------------------------


def test_compute_test_changes_is_expc_minus_original():
    original = _metrics(
        0.35,
        0.96,
        0.67,
        {
            "N": _per_class(0.99, 0.97, 0.98, 13000),
            "S": _per_class(0.80, 0.78, 0.79, 300),
            "V": _per_class(0.94, 0.92, 0.93, 1000),
            "F": _per_class(0.02, 0.05, 0.00, 20),
        },
    )
    expc = _metrics(
        0.33,
        0.965,
        0.70,
        {
            "N": _per_class(0.99, 0.97, 0.98, 13000),
            "S": _per_class(0.83, 0.80, 0.82, 300),
            "V": _per_class(0.94, 0.93, 0.94, 1000),
            "F": _per_class(0.05, 0.08, 0.06, 20),
        },
    )

    # The implementation returns a JSON-compatible dictionary. This cast
    # gives Pylance the precise nested structure expected by this test.
    change = cast(
        ChangesResult,
        compute_test_changes(original, expc),
    )

    assert change["overall"]["macro_f1"] == pytest.approx(0.03)
    assert change["overall"]["accuracy"] == pytest.approx(0.005)
    assert change["overall"]["loss"] == pytest.approx(-0.02)
    assert change["per_class"]["S"]["f1"] == pytest.approx(0.03)
    assert change["per_class"]["S"]["support"] == 300
    assert change["per_class"]["F"]["f1"] == pytest.approx(0.06)


# ---------------------------------------------------------------------
#              Prediction Agreement / Correctness Transitions
# ---------------------------------------------------------------------


def test_prediction_agreement_and_transitions():
    true = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    original_pred = np.array([0, 1, 2, 0, 2], dtype=np.int64)
    expc_pred = np.array([0, 2, 2, 1, 1], dtype=np.int64)

    agreement = compute_prediction_agreement(
        original_pred,
        expc_pred,
    )

    assert agreement["num_targets"] == 5
    assert agreement["identical_count"] == 2
    assert agreement["changed_count"] == 3
    assert agreement["identical_fraction"] == pytest.approx(0.4)

    transitions = compute_correctness_transitions(
        true,
        original_pred,
        expc_pred,
    )

    assert transitions["correct_to_correct"] == 2
    assert transitions["correct_to_incorrect"] == 2
    assert transitions["incorrect_to_correct"] == 1
    assert transitions["incorrect_to_incorrect"] == 0

    # The four transition counts must partition every paired target.
    assert sum(transitions.values()) == true.size


# ---------------------------------------------------------------------
#                     SHA-256 And Result Schema
# ---------------------------------------------------------------------


def test_sha256_file_is_deterministic(tmp_path: Path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"deterministic-content")

    expected = hashlib.sha256(b"deterministic-content").hexdigest()
    assert sha256_file(path) == expected
    assert sha256_file(path) == sha256_file(path)


def test_build_test_comparison_has_only_intended_sections():
    comparison = cast(
        ComparisonResult,
        build_test_comparison(
            test_dataset_dir=Path("data/splits_sequences_xqrs/test"),
            test_dataset_summary={"split_name": "test"},
            configuration={"num_layers": 3},
            original_checkpoint_path=Path("original.pt"),
            original_sha256="a",
            original_metrics_json={"loss": 0.0},
            expc_checkpoint_path=Path("expc.pt"),
            expc_sha256="b",
            expc_metrics_json={"loss": 0.0},
            change={"overall": {}, "per_class": {}},
            prediction_agreement={"num_targets": 0},
            correctness_transitions={"correct_to_correct": 0},
        ),
    )

    assert set(comparison) == {
        "evaluation_type",
        "test_dataset_dir",
        "test_dataset_summary",
        "configuration",
        "original",
        "expc",
        "change",
        "prediction_agreement",
        "correctness_transitions",
    }
    assert comparison["evaluation_type"] == ("final_xqrs_centered_test_comparison")
    assert set(comparison["original"]) == {
        "checkpoint_path",
        "checkpoint_sha256",
        "metrics",
    }
    assert comparison["expc"]["checkpoint_sha256"] == "b"
