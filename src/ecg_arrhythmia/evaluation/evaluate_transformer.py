import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.ecg_sequence_dataset import ECGSequenceDataset
from ecg_arrhythmia.data.label_mapping import CLASS_LABELS, NUM_CLASSES
from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer
from ecg_arrhythmia.training.transformer_training import (
    EvaluationMetrics,
    compute_class_weights,
    evaluate,
    log_confusion_matrix,
    log_per_class_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#                    Helpers For Evaluate Logic
# ---------------------------------------------------------------------


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_layers: int,
) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")

    # Create an empty model with the same architecture used during training.
    model = ECGSequenceTransformer(
        num_classes=NUM_CLASSES,
        num_layers=num_layers,
    ).to(device)

    # Load the saved weights onto whichever device is currently available.
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


def build_confusion_matrix_rows(
    confusion_matrix: list[list[int]],
) -> list[dict[str, object]]:
    return [
        {
            "true_label": true_label,
            "predictions": dict(zip(CLASS_LABELS, row, strict=True)),
            "total": sum(row),
        }
        for true_label, row in zip(
            CLASS_LABELS,
            confusion_matrix,
            strict=True,
        )
    ]


def format_metrics_for_json(
    metrics: EvaluationMetrics,
) -> dict[str, object]:
    return {
        "loss": round(metrics["loss"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "per_class": metrics["per_class"],
        "confusion_matrix": {
            "description": "Rows are true labels; columns are predicted labels.",
            "labels": CLASS_LABELS,
            "rows": build_confusion_matrix_rows(metrics["confusion_matrix"]),
        },
    }


def save_metrics(
    metrics: EvaluationMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf8") as file:
        json.dump(format_metrics_for_json(metrics), file, indent=4)


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the CNN + RR Transformer on the sequence test set."
    )

    # Add our command line arguments
    parser.add_argument(
        "--test-split-dir",
        type=Path,
        default=Path("data/splits_sequences/test"),
    )

    parser.add_argument(
        "--train-split-dir",
        type=Path,
        default=Path("data/splits_sequences/train"),
        help=(
            "Training split used to recreate the same class weights "
            "used during training."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/models/ecg_sequence_transformer.pt"),
        help="Path to the saved Transformer model weights.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/results/ecg_sequence_transformer_test_metrics.json"),
        help="Where to save the test evaluation metrics.",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers used by the checkpoint.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
#                       Main Evaluate Logic
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Use a GPU if CUDA is available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        logger.info("device: CUDA | %s", torch.cuda.get_device_name(0))
    else:
        logger.info("device: %s", device)

    # Load both datasets. The training set is only needed to recreate
    # the class weights used by the weighted CrossEntropyLoss.
    train_set = ECGSequenceDataset(args.train_split_dir)
    test_set = ECGSequenceDataset(args.test_split_dir)

    # Test data must not be shuffled so evaluation is deterministic.
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Build the model skeleton and populate it with the saved best weights.
    model = load_model(
        checkpoint_path=args.checkpoint_path, device=device, num_layers=args.num_layers
    )

    # Use the same weighted loss setup as training, based only on
    # the target-label counts in the training sequence split.
    class_weights = compute_class_weights(train_set, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info("checkpoint path: %s", args.checkpoint_path)
    logger.info("class weights: %s", class_weights)

    # Evaluate the saved model on the held-out sequence test set.
    metrics = evaluate(
        model=model,
        split_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    logger.info(
        "test loss: %.4f | test acc: %.4f | test macro f1: %.4f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )

    log_per_class_metrics(metrics)
    log_confusion_matrix(metrics)

    save_metrics(
        metrics=metrics,
        output_path=args.output_path,
    )

    logger.info("saved test metrics to: %s", args.output_path)


if __name__ == "__main__":
    main()
