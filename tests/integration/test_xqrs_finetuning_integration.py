import json
import sys
from unittest.mock import patch

import numpy as np
import torch

from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer
from ecg_arrhythmia.training import transformer_training as training

WINDOW_SIZE = 240
SEQUENCE_LENGTH = 5
CLASSES = ["N", "S", "V", "F"]


def _make_sequence_split(directory, num_sequences, seed):
    """Write a tiny synthetic sequence split covering all four classes."""

    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)

    X = rng.standard_normal((num_sequences, SEQUENCE_LENGTH, WINDOW_SIZE)).astype(
        np.float64
    )
    rr = rng.uniform(0.5, 1.5, size=(num_sequences, SEQUENCE_LENGTH, 2)).astype(
        np.float64
    )
    labels = np.array([CLASSES[i % len(CLASSES)] for i in range(num_sequences)])

    np.save(directory / "X.npy", X)
    np.save(directory / "rr_features.npy", rr)
    np.save(directory / "y.npy", labels)


def test_finetuning_runs_and_writes_summary(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_sequence_split(train_dir, num_sequences=12, seed=1)
    _make_sequence_split(val_dir, num_sequences=8, seed=2)

    # A tiny initial checkpoint matching the requested architecture.
    torch.manual_seed(0)
    initial_model = ECGSequenceTransformer(num_classes=4, num_layers=1, dropout=0.3)
    checkpoint_path = tmp_path / "initial.pt"
    torch.save(initial_model.state_dict(), checkpoint_path)

    output_path = tmp_path / "finetuned.pt"
    summary_path = tmp_path / "finetuning_summary.json"

    argv = [
        "transformer_training",
        "--train-set-dir",
        str(train_dir),
        "--val-set-dir",
        str(val_dir),
        "--initial-checkpoint-path",
        str(checkpoint_path),
        "--model-output-path",
        str(output_path),
        "--finetuning-summary-path",
        str(summary_path),
        "--num-layers",
        "1",
        "--dropout",
        "0.3",
        "--learning-rate",
        "0.0001",
        "--epochs",
        "2",
        "--patience",
        "5",
        "--batch-size",
        "4",
        "--seed",
        "42",
        "--class-weighting",
        "inverse",
    ]

    with patch.object(sys, "argv", argv):
        training.main()

    # The candidate checkpoint is written (at least the baseline, saved first).
    assert output_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["initial_checkpoint_path"] == str(checkpoint_path)
    assert summary["finetuned_checkpoint_path"] == str(output_path)
    assert summary["train_set_dir"] == str(train_dir)
    assert summary["validation_set_dir"] == str(val_dir)

    # Baseline evaluation happened and is stored compactly (no confusion matrix).
    assert set(summary["baseline"]) == {"loss", "accuracy", "macro_f1", "per_class"}
    assert "confusion_matrix" not in summary["baseline"]
    assert set(summary["change"]) == {"loss", "accuracy", "macro_f1", "per_class_f1"}
    assert set(summary["change"]["per_class_f1"]) == set(CLASSES)

    # At least one epoch ran.
    assert summary["training"]["epochs_completed"] >= 1

    # The written checkpoint loads strictly into the same architecture.
    reloaded = ECGSequenceTransformer(num_classes=4, num_layers=1, dropout=0.3)
    reloaded.load_state_dict(torch.load(output_path, map_location="cpu"), strict=True)
