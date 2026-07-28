import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.build_xqrs_centered_dataset import (
    assert_splits_pairwise_disjoint,
    build_and_save_xqrs_centered_dataset,
    load_split_record_names,
)
from ecg_arrhythmia.data.ecg_sequence_dataset import ECGSequenceDataset
from ecg_arrhythmia.evaluation.evaluate_transformer import format_metrics_for_json
from ecg_arrhythmia.evaluation.paired_centering_comparison import (
    compute_correctness_transitions,
    compute_prediction_agreement,
)
from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer
from ecg_arrhythmia.training.transformer_training import (
    CLASS_LABELS,
    NUM_CLASSES,
    EvaluationMetrics,
    compute_class_weights,
    evaluate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#                         Default Paths / Config
# ---------------------------------------------------------------------

SPLIT_SUMMARY_PATH = Path("data/splits_sequences_matched/split_summary_metrics.json")
XQRS_TRAIN_DIR = Path("data/splits_sequences_xqrs/train")
XQRS_TEST_DIR = Path("data/splits_sequences_xqrs/test")

ORIGINAL_CHECKPOINT_PATH = Path("artifacts/models/ecg_sequence_transformer_tuned.pt")
EXPC_CHECKPOINT_PATH = Path("artifacts/models/ecg_sequence_transformer_xqrs_EXPC.pt")

RESULT_PATH = Path(
    "artifacts/results/model_evaluation/transformer_xqrs_test_comparison.json"
)

TUNED_NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 64
MATCHING_TOLERANCE_MS = 100.0

# The EXPC checkpoint was selected with capped inverse class weights, so the
# shared comparison criterion uses the same weighting.
CLASS_WEIGHTING = "capped_inverse"
MAX_CLASS_WEIGHT = 5.0

# Files that must exist for the test dataset to be reused without rebuilding.
REQUIRED_TEST_FILES = ("X.npy", "y.npy", "rr_features.npy", "dataset_summary.json")


# ---------------------------------------------------------------------
#                              Helpers
# ---------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(
    checkpoint_path: Path,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    """
    Build a model skeleton and load a checkpoint into it strictly.

    A missing checkpoint raises ``FileNotFoundError``; ``strict=True`` makes
    any architecture mismatch fail loudly.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model = ECGSequenceTransformer(
        num_classes=NUM_CLASSES,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Return the true and predicted class indices in dataset order.

    The loader must not be shuffled so the arrays align row-for-row with the
    saved sequences.
    """

    model.eval()

    true_chunks: list[NDArray[np.int64]] = []
    predicted_chunks: list[NDArray[np.int64]] = []

    with torch.inference_mode():
        for X_batch, rr_batch, y_batch in loader:
            logits = model(X_batch.to(device), rr_batch.to(device))
            predictions = logits.argmax(dim=1)

            true_chunks.append(y_batch.numpy().astype(np.int64))
            predicted_chunks.append(predictions.cpu().numpy().astype(np.int64))

    return np.concatenate(true_chunks), np.concatenate(predicted_chunks)


def compute_test_changes(
    original_metrics: EvaluationMetrics,
    expc_metrics: EvaluationMetrics,
) -> dict[str, float]:
    """Absolute EXPC-minus-original change in the test metrics."""

    overall = {
        "loss": round(expc_metrics["loss"] - original_metrics["loss"], 4),
        "accuracy": round(expc_metrics["accuracy"] - original_metrics["accuracy"], 4),
        "macro_f1": round(expc_metrics["macro_f1"] - original_metrics["macro_f1"], 4),
    }

    per_class: dict[str, dict[str, object]] = {}
    for label in CLASS_LABELS:
        original_class = original_metrics["per_class"][label]
        expc_class = expc_metrics["per_class"][label]
        per_class[label] = {
            "precision": round(
                expc_class["precision"] - original_class["precision"], 4
            ),
            "recall": round(expc_class["recall"] - original_class["recall"], 4),
            "f1": round(expc_class["f1"] - original_class["f1"], 4),
            "support": expc_class["total_class_count"],
        }

    return {"overall": overall, "per_class": per_class}


def build_test_comparison(
    test_dataset_dir: Path,
    test_dataset_summary: dict[str, object],
    configuration: dict[str, object],
    original_checkpoint_path: Path,
    original_sha256: str,
    original_metrics_json: dict[str, object],
    expc_checkpoint_path: Path,
    expc_sha256: str,
    expc_metrics_json: dict[str, object],
    change: dict[str, object],
    prediction_agreement: dict[str, object],
    correctness_transitions: dict[str, int],
) -> dict[str, object]:
    """Assemble the single authoritative test-comparison result."""

    return {
        "evaluation_type": "final_xqrs_centered_test_comparison",
        "test_dataset_dir": str(test_dataset_dir),
        "test_dataset_summary": test_dataset_summary,
        "configuration": configuration,
        "original": {
            "checkpoint_path": str(original_checkpoint_path),
            "checkpoint_sha256": original_sha256,
            "metrics": original_metrics_json,
        },
        "expc": {
            "checkpoint_path": str(expc_checkpoint_path),
            "checkpoint_sha256": expc_sha256,
            "metrics": expc_metrics_json,
        },
        "change": change,
        "prediction_agreement": prediction_agreement,
        "correctness_transitions": correctness_transitions,
    }


def ensure_test_dataset(
    test_dir: Path,
    summary_path: Path,
    tolerance_ms: float,
    normalise_beats: bool,
    rebuild: bool,
) -> None:
    """
    Build the XQRS-centred test dataset if missing, or reuse it otherwise.

    Train and validation are never rebuilt here. The split record sets are
    proven pairwise disjoint before building.
    """

    ready = all((test_dir / name).exists() for name in REQUIRED_TEST_FILES)
    if rebuild or not ready:
        assert_splits_pairwise_disjoint(summary_path)
        record_names = load_split_record_names(summary_path, "test")
        logger.info(
            "Building XQRS-centred test dataset (%d records) at %s",
            len(record_names),
            test_dir,
        )
        build_and_save_xqrs_centered_dataset(
            record_names=record_names,
            output_dir=test_dir,
            tolerance_ms=tolerance_ms,
            normalise_beats=normalise_beats,
            split_name="test",
        )
    else:
        logger.info("Reusing existing XQRS-centred test dataset at %s", test_dir)


def load_test_dataset_summary(test_dir: Path) -> dict[str, object]:
    """Read the compact test summary written by the dataset builder."""

    with (test_dir / "dataset_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)

    record_names = summary["record_names"]
    return {
        "split_name": summary["split_name"],
        "record_names": record_names,
        "num_records": len(record_names),
        "num_final_sequences": summary["num_final_sequences"],
        "target_class_distribution": summary["target_class_distribution"],
    }


# ---------------------------------------------------------------------
#                            Printed Summary
# ---------------------------------------------------------------------


def print_summary(
    original_metrics: EvaluationMetrics,
    expc_metrics: EvaluationMetrics,
    test_dataset_summary: dict[str, object],
    num_targets: int,
    agreement: dict[str, object],
    transitions: dict[str, int],
) -> None:
    """Print a concise final comparison."""

    print("\nFinal XQRS-centred test comparison\n")
    print(f"Test records: {test_dataset_summary['num_records']}")
    print(f"Test targets: {num_targets}\n")

    print(f"{'Metric':<16}{'Original':>12}{'EXPC':>12}{'Change':>12}")
    print("-" * 52)
    rows = (
        ("Loss", "loss"),
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
    )
    for name, key in rows:
        original_value = original_metrics[key]
        expc_value = expc_metrics[key]
        print(
            f"{name:<16}{original_value:>12.4f}{expc_value:>12.4f}"
            f"{expc_value - original_value:>+12.4f}"
        )

    print(f"\n{'Class':<8}{'Original F1':>14}{'EXPC F1':>12}{'Change':>12}")
    print("-" * 46)
    for label in CLASS_LABELS:
        original_f1 = original_metrics["per_class"][label]["f1"]
        expc_f1 = expc_metrics["per_class"][label]["f1"]
        print(
            f"{label:<8}{original_f1:>14.4f}{expc_f1:>12.4f}"
            f"{expc_f1 - original_f1:>+12.4f}"
        )

    print(
        f"\nPrediction agreement: {agreement['identical_count']}/"
        f"{agreement['num_targets']} ({agreement['identical_fraction']})"
    )
    print(f"Correct -> incorrect: {transitions['correct_to_incorrect']}")
    print(f"Incorrect -> correct: {transitions['incorrect_to_correct']}")

    improved = expc_metrics["macro_f1"] > original_metrics["macro_f1"]
    print(f"\nEXPC improved test macro F1 over the original checkpoint: {improved}")


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final locked XQRS-centred test comparison of the original tuned "
            "checkpoint and the selected EXPC fine-tuned checkpoint."
        )
    )

    parser.add_argument("--split-summary-path", type=Path, default=SPLIT_SUMMARY_PATH)
    parser.add_argument("--xqrs-train-dir", type=Path, default=XQRS_TRAIN_DIR)
    parser.add_argument("--xqrs-test-dir", type=Path, default=XQRS_TEST_DIR)
    parser.add_argument(
        "--original-checkpoint-path", type=Path, default=ORIGINAL_CHECKPOINT_PATH
    )
    parser.add_argument(
        "--expc-checkpoint-path", type=Path, default=EXPC_CHECKPOINT_PATH
    )
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--num-layers", type=int, default=TUNED_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=MATCHING_TOLERANCE_MS)
    parser.add_argument("--normalise-beats", action="store_true")
    parser.add_argument(
        "--rebuild-test",
        action="store_true",
        help="Rebuild the XQRS-centred test dataset even if it already exists.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
#                                 Main
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    # Build or reuse the deployment-realistic XQRS-centred test dataset.
    ensure_test_dataset(
        test_dir=args.xqrs_test_dir,
        summary_path=args.split_summary_path,
        tolerance_ms=args.tolerance_ms,
        normalise_beats=args.normalise_beats,
        rebuild=args.rebuild_test,
    )

    test_dataset = ECGSequenceDataset(args.xqrs_test_dir)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # One fixed criterion for both checkpoints: capped-inverse weights from the
    # XQRS-centred training split. Loss is only comparable because it is shared.
    train_set = ECGSequenceDataset(args.xqrs_train_dir)
    class_weights = compute_class_weights(
        dataset=train_set,
        device=device,
        weighting_method=CLASS_WEIGHTING,
        max_class_weight=MAX_CLASS_WEIGHT,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Load both frozen checkpoints strictly. Neither is trained or modified.
    original_model = load_checkpoint_model(
        args.original_checkpoint_path, args.num_layers, args.dropout, device
    )
    expc_model = load_checkpoint_model(
        args.expc_checkpoint_path, args.num_layers, args.dropout, device
    )

    # Evaluate both on the identical test loader.
    original_metrics = evaluate(original_model, test_loader, criterion, device)
    original_true, original_pred = collect_predictions(
        original_model, test_loader, device
    )
    expc_metrics = evaluate(expc_model, test_loader, criterion, device)
    expc_true, expc_pred = collect_predictions(expc_model, test_loader, device)

    # Alignment guarantees.
    num_targets = len(test_dataset)
    if not np.array_equal(original_true, expc_true):
        raise ValueError("Test true labels differ between the two passes.")
    if original_pred.shape[0] != num_targets or expc_pred.shape[0] != num_targets:
        raise ValueError("Predictions are not aligned with the test dataset.")
    for label in CLASS_LABELS:
        original_support = original_metrics["per_class"][label]["total_class_count"]
        expc_support = expc_metrics["per_class"][label]["total_class_count"]
        if original_support != expc_support:
            raise ValueError(f"Class support differs for {label}.")

    true_labels = original_true
    agreement = compute_prediction_agreement(original_pred, expc_pred)
    transitions = compute_correctness_transitions(true_labels, original_pred, expc_pred)
    if sum(transitions.values()) != num_targets:
        raise ValueError("Correctness transitions do not sum to the target count.")

    change = compute_test_changes(original_metrics, expc_metrics)

    configuration = {
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "class_weighting": CLASS_WEIGHTING,
        "max_class_weight": MAX_CLASS_WEIGHT,
    }

    comparison = build_test_comparison(
        test_dataset_dir=args.xqrs_test_dir,
        test_dataset_summary=load_test_dataset_summary(args.xqrs_test_dir),
        configuration=configuration,
        original_checkpoint_path=args.original_checkpoint_path,
        original_sha256=sha256_file(args.original_checkpoint_path),
        original_metrics_json=format_metrics_for_json(original_metrics),
        expc_checkpoint_path=args.expc_checkpoint_path,
        expc_sha256=sha256_file(args.expc_checkpoint_path),
        expc_metrics_json=format_metrics_for_json(expc_metrics),
        change=change,
        prediction_agreement=agreement,
        correctness_transitions=transitions,
    )

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    with args.result_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    print_summary(
        original_metrics=original_metrics,
        expc_metrics=expc_metrics,
        test_dataset_summary=comparison["test_dataset_summary"],
        num_targets=num_targets,
        agreement=agreement,
        transitions=transitions,
    )

    logger.info("Saved final test comparison to %s", args.result_path)


if __name__ == "__main__":
    main()
