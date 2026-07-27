import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ecg_arrhythmia.evaluation import evaluate_paired_xqrs_validation as paired_runner

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_SUMMARY = REPO_ROOT / "data/splits_sequences_matched/split_summary_metrics.json"
EXPERT_VAL_DIR = REPO_ROOT / "data/splits_sequences_matched/val"
TRAIN_DIR = REPO_ROOT / "data/splits_sequences_matched/train"
XQRS_VAL_DIR = REPO_ROOT / "data/splits_sequences_xqrs/val"
CHECKPOINT_PATH = REPO_ROOT / "artifacts/models/ecg_sequence_transformer_tuned.pt"

EXPECTED_TOP_LEVEL_KEYS = {
    "checkpoint_path",
    "num_layers",
    "pairing_summary",
    "expert_centered",
    "xqrs_centered",
    "change",
    "prediction_agreement",
    "correctness_transitions",
}

pytestmark = pytest.mark.skipif(
    not (
        SPLIT_SUMMARY.exists()
        and EXPERT_VAL_DIR.exists()
        and TRAIN_DIR.exists()
        and (XQRS_VAL_DIR / "audit_records.npy").exists()
        and CHECKPOINT_PATH.exists()
    ),
    reason="Matched split, XQRS-centred dataset, and tuned checkpoint required.",
)


def test_paired_runner_writes_single_compact_json(tmp_path):
    comparison_path = tmp_path / "transformer_paired_centering_comparison.json"

    argv = [
        "evaluate_paired_xqrs_validation",
        "--split-summary-path",
        str(SPLIT_SUMMARY),
        "--expert-val-dir",
        str(EXPERT_VAL_DIR),
        "--train-dir",
        str(TRAIN_DIR),
        "--xqrs-val-dir",
        str(XQRS_VAL_DIR),
        "--checkpoint-path",
        str(CHECKPOINT_PATH),
        "--paired-expert-dir",
        str(tmp_path / "paired_expert"),
        "--paired-xqrs-dir",
        str(tmp_path / "paired_xqrs"),
        "--comparison-path",
        str(comparison_path),
        "--device",
        "cpu",
    ]

    with patch.object(sys, "argv", argv):
        paired_runner.main()

    # Exactly one result file, with the compact schema.
    assert comparison_path.exists()
    result = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert set(result) == EXPECTED_TOP_LEVEL_KEYS

    # No confusion matrices or removed diagnostics remain.
    assert "confusion_matrix" not in result["expert_centered"]
    assert "confusion_matrix" not in result["xqrs_centered"]
    assert set(result["change"]) == {"overall_change", "per_class_f1_change"}
    assert set(result["prediction_agreement"]) == {
        "num_targets",
        "identical_count",
        "identical_fraction",
        "changed_count",
    }
    assert set(result["correctness_transitions"]) == {
        "correct_to_correct",
        "correct_to_incorrect",
        "incorrect_to_correct",
        "incorrect_to_incorrect",
    }

    # Paired counts and identical supports across both conditions.
    pairing = result["pairing_summary"]
    assert pairing["paired_targets"] > 0
    expert_per_class = result["expert_centered"]["per_class"]
    xqrs_per_class = result["xqrs_centered"]["per_class"]
    for label, count in pairing["paired_class_support"].items():
        assert expert_per_class[label]["total_class_count"] == count
        assert xqrs_per_class[label]["total_class_count"] == count

    # No separate metric JSON files or prediction NPZ were produced.
    assert not list(tmp_path.rglob("*.npz"))
    assert not list(tmp_path.rglob("*expert_centered_validation.json"))
    assert not list(tmp_path.rglob("*xqrs_centered_validation.json"))
