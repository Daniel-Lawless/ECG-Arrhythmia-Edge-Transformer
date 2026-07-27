import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.build_expert_centered_dataset import (
    build_expert_centered_dataset,
    verify_matches_saved_split,
)
from ecg_arrhythmia.data.build_xqrs_centered_dataset import (
    build_and_save_xqrs_centered_dataset,
)
from ecg_arrhythmia.data.ecg_sequence_dataset import ECGSequenceDataset
from ecg_arrhythmia.evaluation.evaluate_r_peak_validation import (
    EXPECTED_VALIDATION_RECORDS,
    load_validation_record_names,
)
from ecg_arrhythmia.evaluation.evaluate_transformer import load_model
from ecg_arrhythmia.evaluation.paired_centering_comparison import (
    compare_paired_metrics,
    compute_correctness_transitions,
    compute_prediction_agreement,
)
from ecg_arrhythmia.evaluation.paired_target_index import (
    build_paired_target_index,
    build_pairing_summary,
    create_paired_dataset_views,
)
from ecg_arrhythmia.training.transformer_training import (
    EvaluationMetrics,
    compute_class_weights,
    evaluate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#                         Default Paths / Config
# ---------------------------------------------------------------------

SPLIT_SUMMARY_PATH = Path("data/splits_sequences_matched/split_summary_metrics.json")
EXPERT_VAL_DIR = Path("data/splits_sequences_matched/val")
TRAIN_DIR = Path("data/splits_sequences_matched/train")
XQRS_VAL_DIR = Path("data/splits_sequences_xqrs/val")

PAIRED_EXPERT_DIR = Path("data/splits_sequences_paired/expert_centered")
PAIRED_XQRS_DIR = Path("data/splits_sequences_paired/xqrs_centered")

CHECKPOINT_PATH = Path("artifacts/models/ecg_sequence_transformer_tuned.pt")
TUNED_NUM_LAYERS = 3

RESULTS_DIR = Path("artifacts/results/model_evaluation")
PAIRED_COMPARISON_PATH = RESULTS_DIR / "transformer_paired_centering_comparison.json"

MATCHING_TOLERANCE_MS = 100.0

# The audit arrays are needed to identify the expert heartbeat behind each
# XQRS-centred target and to inspect the detector offset and sequence context.
REQUIRED_XQRS_FILES = (
    "X.npy",
    "y.npy",
    "rr_features.npy",
    "audit_records.npy",
    "audit_annotation_samples.npy",
    "audit_offset_samples.npy",
    "audit_offset_ms.npy",
    "audit_has_unmatched_context.npy",
)


# ---------------------------------------------------------------------
#                              IO Helpers
# ---------------------------------------------------------------------


def load_xqrs_centered_dataset(xqrs_val_dir: Path) -> dict[str, np.ndarray]:
    """Load the XQRS-centred sequences and audit arrays."""

    # Fail before loading anything so a partial dataset cannot be evaluated.
    missing = [
        name for name in REQUIRED_XQRS_FILES if not (xqrs_val_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing XQRS-centred files {missing} in {xqrs_val_dir}. "
            "Re-run with --rebuild-xqrs to build them."
        )

    # Cast the audit fields to stable dtypes because they are used for exact
    # identity and alignment checks later in the comparison.
    return {
        "X": np.load(xqrs_val_dir / "X.npy"),
        "y": np.load(xqrs_val_dir / "y.npy").astype(str),
        "rr": np.load(xqrs_val_dir / "rr_features.npy"),
        "records": np.load(xqrs_val_dir / "audit_records.npy").astype(str),
        "annotation_samples": np.load(
            xqrs_val_dir / "audit_annotation_samples.npy"
        ).astype(np.int64),
        "offset_samples": np.load(xqrs_val_dir / "audit_offset_samples.npy").astype(
            np.int64
        ),
        "offset_ms": np.load(xqrs_val_dir / "audit_offset_ms.npy").astype(np.float64),
        "has_unmatched_context": np.load(
            xqrs_val_dir / "audit_has_unmatched_context.npy"
        ).astype(bool),
    }


def save_sequence_arrays(
    output_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    rr: np.ndarray,
) -> None:
    """Save aligned sequence arrays in the format ECGSequenceDataset expects."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "rr_features.npy", rr)


def ensure_xqrs_dataset(
    xqrs_val_dir: Path,
    record_names: list[str],
    tolerance_ms: float,
    normalise_beats: bool,
    rebuild: bool,
) -> None:
    """
    Build the XQRS-centred dataset if it is missing or a rebuild is
    requested, reusing the existing arrays otherwise.

    The XQRS windows, RR features, false-positive context and audit arrays
    are produced entirely by ``build_and_save_xqrs_centered_dataset``; this
    function does not change how they are built.
    """

    # Reuse the saved build unless one of its required outputs is missing.
    ready = all((xqrs_val_dir / name).exists() for name in REQUIRED_XQRS_FILES)
    if rebuild or not ready:
        logger.info("Building XQRS-centred validation dataset at %s", xqrs_val_dir)
        build_and_save_xqrs_centered_dataset(
            record_names=record_names,
            output_dir=xqrs_val_dir,
            tolerance_ms=tolerance_ms,
            normalise_beats=normalise_beats,
        )
    else:
        logger.info("Reusing existing XQRS-centred dataset at %s", xqrs_val_dir)


# ---------------------------------------------------------------------
#                          Inference Helpers
# ---------------------------------------------------------------------


def evaluate_split(
    split_dir: Path,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[EvaluationMetrics, int]:
    """Evaluate the model on one sequence split using the shared metric."""

    dataset = ECGSequenceDataset(split_dir)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Both conditions use the same dataset class, loss and evaluation code.
    metrics = evaluate(
        model=model,
        split_loader=loader,
        criterion=criterion,
        device=device,
    )

    return metrics, len(dataset)


def collect_predictions(
    split_dir: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Run inference and return the true and predicted class indices in
    dataset order.

    The loader is not shuffled, so the returned arrays align row-for-row
    with the saved sequences.
    """

    dataset = ECGSequenceDataset(split_dir)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    true_chunks: list[NDArray[np.int64]] = []
    predicted_chunks: list[NDArray[np.int64]] = []

    # Gradients are not needed, and keeping the loader unshuffled preserves
    # the target-by-target pairing between the two conditions.
    with torch.inference_mode():
        for X_batch, rr_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            rr_batch = rr_batch.to(device)

            logits = model(X_batch, rr_batch)
            # The largest logit gives the predicted AAMI class index.
            predictions = logits.argmax(dim=1)

            true_chunks.append(y_batch.numpy().astype(np.int64))
            predicted_chunks.append(predictions.cpu().numpy().astype(np.int64))

    return np.concatenate(true_chunks), np.concatenate(predicted_chunks)


def _metrics_summary(metrics: EvaluationMetrics) -> dict[str, object]:
    """Compact per-condition metric summary (no confusion matrix)."""

    return {
        "loss": round(metrics["loss"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "per_class": metrics["per_class"],
    }


# ---------------------------------------------------------------------
#                            Printed Summary
# ---------------------------------------------------------------------


def print_summary(
    expert_metrics: EvaluationMetrics,
    xqrs_metrics: EvaluationMetrics,
    agreement: dict[str, object],
    transitions: dict[str, int],
    paired_count: int,
) -> None:
    """Print a concise paired comparison."""

    print(f"\nPaired matched-target comparison ({paired_count} targets)\n")
    print(f"{'Metric':<16}{'Expert':>12}{'XQRS':>12}{'Change':>12}")
    print("-" * 52)
    metric_rows = (("Loss", "loss"), ("Accuracy", "accuracy"), ("Macro F1", "macro_f1"))
    for name, key in metric_rows:
        expert_value = expert_metrics[key]
        xqrs_value = xqrs_metrics[key]
        print(
            f"{name:<16}{expert_value:>12.4f}{xqrs_value:>12.4f}"
            f"{xqrs_value - expert_value:>+12.4f}"
        )

    print(
        f"\nPrediction agreement: {agreement['identical_count']}/"
        f"{agreement['num_targets']} "
        f"({agreement['identical_fraction']:.4f})"
    )
    print(
        "Correctness transitions | "
        f"correct->incorrect: {transitions['correct_to_incorrect']} | "
        f"incorrect->correct: {transitions['incorrect_to_correct']}"
    )

    print(f"\n{'Class':<8}{'Expert F1':>12}{'XQRS F1':>12}{'Change':>12}")
    print("-" * 44)
    for label in xqrs_metrics["per_class"]:
        expert_f1 = expert_metrics["per_class"][label]["f1"]
        xqrs_f1 = xqrs_metrics["per_class"][label]["f1"]
        print(
            f"{label:<8}{expert_f1:>12.4f}{xqrs_f1:>12.4f}{xqrs_f1 - expert_f1:>+12.4f}"
        )


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired matched-target comparison of the tuned transformer under "
            "expert-centred versus XQRS-centred inputs."
        )
    )

    parser.add_argument("--split-summary-path", type=Path, default=SPLIT_SUMMARY_PATH)
    parser.add_argument("--expert-val-dir", type=Path, default=EXPERT_VAL_DIR)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--xqrs-val-dir", type=Path, default=XQRS_VAL_DIR)
    parser.add_argument("--paired-expert-dir", type=Path, default=PAIRED_EXPERT_DIR)
    parser.add_argument("--paired-xqrs-dir", type=Path, default=PAIRED_XQRS_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--num-layers", type=int, default=TUNED_NUM_LAYERS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tolerance-ms", type=float, default=MATCHING_TOLERANCE_MS)
    parser.add_argument(
        "--normalise-beats",
        action="store_true",
        help="Must match the normalisation used to train the checkpoint.",
    )
    parser.add_argument(
        "--rebuild-xqrs",
        action="store_true",
        help="Rebuild the XQRS-centred dataset even if it already exists.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--comparison-path", type=Path, default=PAIRED_COMPARISON_PATH)

    return parser.parse_args()


# ---------------------------------------------------------------------
#                                 Main
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Use the patient-wise validation records recorded in the split summary.
    record_names = load_validation_record_names(args.split_summary_path)
    logger.info("Validation records: %s", record_names)
    if record_names != EXPECTED_VALIDATION_RECORDS:
        logger.warning(
            "Validation records %s differ from the expected %s.",
            record_names,
            EXPECTED_VALIDATION_RECORDS,
        )

    # Build or reuse the deployment-realistic XQRS-centred dataset, then load it.
    ensure_xqrs_dataset(
        xqrs_val_dir=args.xqrs_val_dir,
        record_names=record_names,
        tolerance_ms=args.tolerance_ms,
        normalise_beats=args.normalise_beats,
        rebuild=args.rebuild_xqrs,
    )
    xqrs = load_xqrs_centered_dataset(args.xqrs_val_dir)

    # Rebuild the expert-centred sequences with identity and confirm they
    # reproduce the saved matched validation split.
    expert_dataset = build_expert_centered_dataset(record_names=record_names)
    verify_matches_saved_split(expert_dataset, args.expert_val_dir)

    # Pair targets by (record, expert annotation sample). The ECG centre may
    # differ between conditions, but the heartbeat being classified must not.
    paired_index = build_paired_target_index(
        expert_records=expert_dataset.target_records,
        expert_annotation_samples=expert_dataset.target_annotation_samples,
        xqrs_records=xqrs["records"],
        xqrs_annotation_samples=xqrs["annotation_samples"],
    )
    if paired_index.num_paired == 0:
        raise ValueError("No paired targets were found between the two pipelines.")

    # Select and order the same targets from both datasets.
    views = create_paired_dataset_views(
        paired_index=paired_index,
        expert_X=expert_dataset.X_sequences,
        expert_rr=expert_dataset.rr_sequences,
        expert_y=expert_dataset.y_labels,
        xqrs_X=xqrs["X"],
        xqrs_rr=xqrs["rr"],
        xqrs_y=xqrs["y"],
        xqrs_records=xqrs["records"],
        xqrs_annotation_samples=xqrs["annotation_samples"],
        xqrs_offset_samples=xqrs["offset_samples"],
        xqrs_offset_ms=xqrs["offset_ms"],
        xqrs_has_unmatched_context=xqrs["has_unmatched_context"],
    )

    # Recreate the target keys from both source datasets rather than trusting
    # the row indices alone. This catches accidental reordering or bad pairing.
    expert_ids = [
        (
            str(expert_dataset.target_records[row]),
            int(expert_dataset.target_annotation_samples[row]),
        )
        for row in paired_index.expert_rows
    ]
    xqrs_ids = [
        (str(xqrs["records"][row]), int(xqrs["annotation_samples"][row]))
        for row in paired_index.xqrs_rows
    ]
    if expert_ids != xqrs_ids or expert_ids != paired_index.paired_keys:
        raise ValueError("Paired target identities are misaligned across conditions.")

    pairing_summary = build_pairing_summary(
        paired_index=paired_index,
        paired_labels=views.expert_y,
    )
    logger.info("Pairing summary: %s", json.dumps(pairing_summary, indent=2))

    # Save the paired subsets in the normal sequence-dataset format. This
    # avoids maintaining a separate inference path just for this comparison.
    save_sequence_arrays(
        args.paired_expert_dir, views.expert_X, views.expert_y, views.expert_rr
    )
    save_sequence_arrays(
        args.paired_xqrs_dir, views.xqrs_X, views.xqrs_y, views.xqrs_rr
    )

    # Load the checkpoint on the requested device and recreate the weighted
    # loss used during training so the reported losses remain comparable.
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    model = load_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        num_layers=args.num_layers,
    )
    train_set = ECGSequenceDataset(args.train_dir)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_set, device))

    # Only the sequence centring differs here; the model, loss and evaluator
    # are identical for the expert-centred and XQRS-centred inputs.
    expert_metrics, expert_count = evaluate_split(
        split_dir=args.paired_expert_dir,
        model=model,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
    )
    xqrs_metrics, xqrs_count = evaluate_split(
        split_dir=args.paired_xqrs_dir,
        model=model,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
    )

    # A paired comparison is invalid if one side contains an extra target.
    if expert_count != xqrs_count:
        raise ValueError(
            f"Paired conditions differ in size: expert={expert_count}, "
            f"xqrs={xqrs_count}."
        )

    # Collect row-level predictions to measure more than the aggregate score:
    # whether each prediction stayed the same, improved or became incorrect.
    expert_true, expert_predictions = collect_predictions(
        split_dir=args.paired_expert_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    xqrs_true, xqrs_predictions = collect_predictions(
        split_dir=args.paired_xqrs_dir,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    if not np.array_equal(expert_true, xqrs_true):
        raise ValueError("Paired expert and XQRS true labels are misaligned.")

    # Agreement ignores correctness; transitions show the direction of any
    # change in correctness caused by using XQRS-centred inputs.
    agreement = compute_prediction_agreement(expert_predictions, xqrs_predictions)
    transitions = compute_correctness_transitions(
        expert_true, expert_predictions, xqrs_predictions
    )
    change = compare_paired_metrics(
        expert_metrics=expert_metrics,
        xqrs_metrics=xqrs_metrics,
        sequence_count=expert_count,
    )

    # Keep the aggregate metrics and paired diagnostics in one result file.
    combined = {
        "checkpoint_path": str(args.checkpoint_path),
        "num_layers": args.num_layers,
        "pairing_summary": pairing_summary,
        "expert_centered": _metrics_summary(expert_metrics),
        "xqrs_centered": _metrics_summary(xqrs_metrics),
        "change": change,
        "prediction_agreement": agreement,
        "correctness_transitions": transitions,
    }

    args.comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with args.comparison_path.open("w", encoding="utf-8") as file:
        json.dump(combined, file, indent=4)

    print_summary(
        expert_metrics=expert_metrics,
        xqrs_metrics=xqrs_metrics,
        agreement=agreement,
        transitions=transitions,
        paired_count=paired_index.num_paired,
    )

    logger.info("Saved paired comparison to %s", args.comparison_path)


if __name__ == "__main__":
    main()
