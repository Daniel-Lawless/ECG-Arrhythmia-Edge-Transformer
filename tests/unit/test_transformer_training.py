from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from ecg_arrhythmia.training.transformer_training import (
    build_finetuning_summary,
    load_initial_checkpoint,
    update_best_checkpoint,
)


def _metrics(loss, accuracy, macro_f1, per_class_f1):
    """Build an EvaluationMetrics-like dict for the summary helpers."""

    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {
            label: {
                "precision": 0.0,
                "recall": 0.0,
                "f1": f1,
                "total_class_count": 10,
            }
            for label, f1 in per_class_f1.items()
        },
        "confusion_matrix": [[0, 0, 0, 0] for _ in range(4)],
    }


# ---------------------------------------------------------------------
#                     Initial Checkpoint Loading
# ---------------------------------------------------------------------


def test_load_initial_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_initial_checkpoint(
            MagicMock(),
            tmp_path / "missing.pt",
            torch.device("cpu"),
        )


def test_load_initial_checkpoint_uses_strict(tmp_path):
    model = MagicMock()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"weight": torch.zeros(2, 2)}, checkpoint_path)

    load_initial_checkpoint(model, checkpoint_path, torch.device("cpu"))

    model.load_state_dict.assert_called_once()
    assert model.load_state_dict.call_args.kwargs.get("strict") is True


# ---------------------------------------------------------------------
#                     Best-Checkpoint Save Rule
# ---------------------------------------------------------------------


def test_update_best_checkpoint_saves_on_improvement(tmp_path):
    model = nn.Linear(2, 2)
    output_path = tmp_path / "candidate.pt"

    best, saved = update_best_checkpoint(0.80, 0.70, model, output_path)

    assert saved is True
    assert best == pytest.approx(0.80)
    assert output_path.exists()


def test_update_best_checkpoint_keeps_baseline_when_not_improved(tmp_path):
    model = nn.Linear(2, 2)
    output_path = tmp_path / "candidate.pt"

    # A baseline checkpoint is saved first, mirroring the fine-tuning flow.
    torch.save(model.state_dict(), output_path)

    best, saved = update_best_checkpoint(0.50, 0.90, model, output_path)

    assert saved is False
    assert best == pytest.approx(0.90)
    assert output_path.exists()


# ---------------------------------------------------------------------
#                       Fine-Tuning Summary
# ---------------------------------------------------------------------


def test_build_finetuning_summary_computes_changes():
    baseline = _metrics(0.32, 0.96, 0.67, {"N": 0.98, "S": 0.79, "V": 0.93, "F": 0.00})
    finetuned = _metrics(0.29, 0.97, 0.71, {"N": 0.98, "S": 0.85, "V": 0.94, "F": 0.10})

    summary = build_finetuning_summary(
        initial_checkpoint_path=Path("artifacts/models/tuned.pt"),
        finetuned_checkpoint_path=Path("artifacts/models/finetuned.pt"),
        train_set_dir=Path("data/splits_sequences_xqrs/train"),
        validation_set_dir=Path("data/splits_sequences_xqrs/val"),
        configuration={"num_layers": 3, "class_weighting": "inverse"},
        baseline_metrics=baseline,
        finetuned_metrics=finetuned,
        epochs_completed=7,
        best_epoch=4,
        stopped_early=True,
    )

    # Compact metric blocks exclude the confusion matrix.
    assert "confusion_matrix" not in summary["baseline"]
    assert set(summary["baseline"]) == {"loss", "accuracy", "macro_f1", "per_class"}

    # Absolute changes are correct.
    assert summary["change"]["macro_f1"] == pytest.approx(0.04)
    assert summary["change"]["accuracy"] == pytest.approx(0.01)
    assert summary["change"]["per_class_f1"]["S"] == pytest.approx(0.06)
    assert summary["change"]["per_class_f1"]["F"] == pytest.approx(0.10)
    assert set(summary["change"]["per_class_f1"]) == {"N", "S", "V", "F"}

    # Paths and training metadata are recorded.
    assert summary["train_set_dir"] == "data/splits_sequences_xqrs/train"
    assert summary["training"] == {
        "epochs_completed": 7,
        "best_epoch": 4,
        "stopped_early": True,
    }
    assert summary["configuration"]["num_layers"] == 3
