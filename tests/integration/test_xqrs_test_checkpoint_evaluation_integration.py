import json
import sys
from unittest.mock import patch

import numpy as np
import torch

from ecg_arrhythmia.evaluation.evaluate_xqrs_test_checkpoints import sha256_file
from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer

WINDOW_SIZE = 240
SEQUENCE_LENGTH = 5
CLASSES = ["N", "S", "V", "F"]


def _make_sequence_split(directory, num_sequences, seed):
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
    return labels


def _write_test_summary(directory, labels):
    distribution = {label: int(np.sum(labels == label)) for label in CLASSES}
    summary = {
        "split_name": "test",
        "record_names": ["100", "103"],
        "num_final_sequences": int(labels.shape[0]),
        "target_class_distribution": distribution,
    }
    (directory / "dataset_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def _save_checkpoint(path, seed):
    torch.manual_seed(seed)
    model = ECGSequenceTransformer(num_classes=4, num_layers=1, dropout=0.3)
    torch.save(model.state_dict(), path)


def _run(tmp_path, result_name, train_dir, test_dir, original_ckpt, expc_ckpt):
    result_path = tmp_path / result_name
    argv = [
        "evaluate_xqrs_test_checkpoints",
        "--xqrs-train-dir",
        str(train_dir),
        "--xqrs-test-dir",
        str(test_dir),
        "--original-checkpoint-path",
        str(original_ckpt),
        "--expc-checkpoint-path",
        str(expc_ckpt),
        "--result-path",
        str(result_path),
        "--num-layers",
        "1",
        "--dropout",
        "0.3",
        "--batch-size",
        "4",
        "--device",
        "cpu",
    ]

    # Import inside the run so we can assert no training loop is invoked.
    from ecg_arrhythmia.evaluation import evaluate_xqrs_test_checkpoints as runner

    with patch.object(sys, "argv", argv):
        runner.main()

    return json.loads(result_path.read_text(encoding="utf-8"))


def test_test_checkpoint_evaluation_end_to_end(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    _make_sequence_split(train_dir, num_sequences=12, seed=1)
    test_labels = _make_sequence_split(test_dir, num_sequences=8, seed=2)
    _write_test_summary(test_dir, test_labels)

    original_ckpt = tmp_path / "original.pt"
    expc_ckpt = tmp_path / "expc.pt"
    _save_checkpoint(original_ckpt, seed=10)
    _save_checkpoint(expc_ckpt, seed=20)

    # Record hashes before running to prove the checkpoints are never modified.
    original_sha_before = sha256_file(original_ckpt)
    expc_sha_before = sha256_file(expc_ckpt)

    result = _run(
        tmp_path, "result1.json", train_dir, test_dir, original_ckpt, expc_ckpt
    )

    # One authoritative result with the intended top-level sections.
    assert set(result) == {
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

    # Both checkpoints were evaluated; their paths and hashes are recorded.
    assert result["original"]["checkpoint_path"] == str(original_ckpt)
    assert result["expc"]["checkpoint_path"] == str(expc_ckpt)
    assert result["original"]["checkpoint_sha256"] == original_sha_before
    assert result["expc"]["checkpoint_sha256"] == expc_sha_before

    # Neither checkpoint file was modified (no training / overwriting).
    assert sha256_file(original_ckpt) == original_sha_before
    assert sha256_file(expc_ckpt) == expc_sha_before

    # Supports are identical across the two models (same test rows).
    original_per_class = result["original"]["metrics"]["per_class"]
    expc_per_class = result["expc"]["metrics"]["per_class"]
    for label in CLASSES:
        assert (
            original_per_class[label]["total_class_count"]
            == expc_per_class[label]["total_class_count"]
        )

    # Correctness transitions partition every test target.
    num_targets = result["prediction_agreement"]["num_targets"]
    assert sum(result["correctness_transitions"].values()) == num_targets

    # A second run produces an identical result (deterministic evaluation).
    result_again = _run(
        tmp_path, "result2.json", train_dir, test_dir, original_ckpt, expc_ckpt
    )
    assert result_again == result
