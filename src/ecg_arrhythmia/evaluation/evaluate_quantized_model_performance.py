import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ecg_arrhythmia.data.build_dataset import EXCLUDED_AAMI_LABELS
from ecg_arrhythmia.data.build_xqrs_centered_dataset import (
    MATCHING_TOLERANCE_MS,
    build_record_detected_beats,
    load_split_record_names,
)
from ecg_arrhythmia.data.label_mapping import (
    CLASS_INDICES,
    CLASS_LABELS,
    LABEL_TO_INDEX,
    NUM_CLASSES,
)
from ecg_arrhythmia.detection.xqrs_detector import XQRSDetector
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import collect_sequences
from ecg_arrhythmia.evaluation.evaluate_quantized_inference_agreement import (
    classify_with_both,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/deployment_evaluation/quantized_model_performance"
)
DEFAULT_FIGURES_DIR = Path("artifacts/figures/quantized_model_performance")
DEFAULT_FP32_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_INT8_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer_int8.onnx")

SUMMARY_FILENAME = "quantized_model_performance_summary.json"
ARRAYS_FILENAME = "quantized_model_performance_arrays.npz"


# ---------------------------------------------------------------------
#                        Ground-Truth Alignment
# ---------------------------------------------------------------------


def resolve_ground_truth(record_name: str) -> dict[int, str]:
    """
    Map each scored target R-peak of a record to its AAMI label.

    Reuses the offline XQRS-centred builder, so labelled sequences use
    the same matching and filtering rules as the deployment dataset.
    """

    conversion = build_record_detected_beats(
        record_name=record_name,
        detector=XQRSDetector(learn=True),
        tolerance_ms=MATCHING_TOLERANCE_MS,
        normalise_beats=False,
        excluded_labels=set(EXCLUDED_AAMI_LABELS),
    )

    return {
        int(peak): str(label)
        for peak, label, is_target in zip(
            conversion.detected_samples,
            conversion.aami_labels,
            conversion.is_target,
            strict=True,
        )
        if is_target
    }


def align_ground_truth(
    target_peaks: list[int],
    labels_by_peak: dict[int, str],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Select the labelled subset of streaming sequences.

    Returns the positions of labelled sequences and their ground-truth
    class indices. Unmatched peaks are excluded from scoring.
    """

    labelled_positions: list[int] = []
    true_indices: list[int] = []

    for position, peak in enumerate(target_peaks):
        label = labels_by_peak.get(int(peak))

        if label is None:
            continue

        if label not in LABEL_TO_INDEX:
            raise ValueError(
                f"Unsupported ground-truth label {label!r} for target peak {peak}."
            )

        labelled_positions.append(position)
        true_indices.append(LABEL_TO_INDEX[label])

    return (
        np.asarray(labelled_positions, dtype=np.int64),
        np.asarray(true_indices, dtype=np.int64),
    )


# ---------------------------------------------------------------------
#                        Classification Metrics
# ---------------------------------------------------------------------


def classification_metrics(
    true_indices: NDArray[np.int64],
    predicted_indices: NDArray[np.int64],
) -> dict:
    """
    Calculate accuracy, macro F1, per-class metrics and confusion matrix.
    """

    true = np.asarray(true_indices, dtype=np.int64)
    predicted = np.asarray(predicted_indices, dtype=np.int64)

    if true.ndim != 1 or true.shape != predicted.shape:
        raise ValueError(
            "Truth and predictions must be matching 1-D arrays, found "
            f"{true.shape} and {predicted.shape}."
        )

    if true.size == 0:
        raise ValueError("At least one labelled sequence is required.")

    report = classification_report(
        true,
        predicted,
        labels=CLASS_INDICES,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true,
        predicted,
        labels=CLASS_INDICES,
    )

    return {
        "num_sequences": int(true.size),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CLASS_LABELS
        },
        "confusion_matrix": matrix.tolist(),
    }


def metric_deltas(fp32_metrics: dict, int8_metrics: dict) -> dict:
    """Signed INT8-minus-FP32 differences for reported metrics."""

    per_class_deltas = {
        label: {
            f"{metric}_delta": (
                int8_metrics["per_class"][label][metric]
                - fp32_metrics["per_class"][label][metric]
            )
            for metric in ("precision", "recall", "f1")
        }
        for label in CLASS_LABELS
    }

    confusion_delta = np.asarray(
        int8_metrics["confusion_matrix"], dtype=np.int64
    ) - np.asarray(fp32_metrics["confusion_matrix"], dtype=np.int64)

    return {
        "accuracy_delta": int8_metrics["accuracy"] - fp32_metrics["accuracy"],
        "macro_f1_delta": int8_metrics["macro_f1"] - fp32_metrics["macro_f1"],
        "per_class_deltas": per_class_deltas,
        "confusion_matrix_delta": confusion_delta.tolist(),
    }


# ---------------------------------------------------------------------
#                     Changed-Prediction Outcomes
# ---------------------------------------------------------------------


def changed_outcomes(
    true_indices: NDArray[np.int64],
    fp32_predictions: NDArray[np.int64],
    int8_predictions: NDArray[np.int64],
) -> dict:
    """
    Score every FP32-INT8 disagreement against ground truth.

    Changed predictions are grouped as harmful, helpful, or both wrong.
    This directly measures whether quantisation's prediction changes
    improve or degrade classification outcomes.
    """

    true = np.asarray(true_indices, dtype=np.int64)
    fp32 = np.asarray(fp32_predictions, dtype=np.int64)
    int8 = np.asarray(int8_predictions, dtype=np.int64)

    if true.ndim != 1 or not (true.shape == fp32.shape == int8.shape):
        raise ValueError(
            "Truth, FP32 and INT8 predictions must be matching 1-D arrays."
        )

    changed = fp32 != int8
    fp32_correct = fp32 == true
    int8_correct = int8 == true

    correct_to_wrong = changed & fp32_correct & ~int8_correct
    wrong_to_correct = changed & ~fp32_correct & int8_correct
    both_wrong = changed & ~fp32_correct & ~int8_correct

    by_class = {
        CLASS_LABELS[index]: {
            "fp32_correct_int8_wrong": int(np.sum(correct_to_wrong & (true == index))),
            "fp32_wrong_int8_correct": int(np.sum(wrong_to_correct & (true == index))),
            "both_wrong": int(np.sum(both_wrong & (true == index))),
        }
        for index in range(NUM_CLASSES)
    }

    harmful = int(np.sum(correct_to_wrong))
    helpful = int(np.sum(wrong_to_correct))

    return {
        "num_changed": int(np.sum(changed)),
        "fp32_correct_int8_wrong": harmful,
        "fp32_wrong_int8_correct": helpful,
        "both_wrong": int(np.sum(both_wrong)),
        "net_correct_change": helpful - harmful,
        "by_ground_truth_class": by_class,
    }


# ---------------------------------------------------------------------
#                            Per-Record Entry
# ---------------------------------------------------------------------


def evaluate_record(
    record_name: str,
    fp32_classifier: ONNXSequenceClassifier,
    int8_classifier: ONNXSequenceClassifier,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[dict, dict[str, NDArray[np.int64]]]:
    """
    Score both models against ground truth for one record.

    Returns a record summary plus labelled arrays used later for exact
    aggregate metrics across all evaluated records.
    """

    # Collect the sequences for this record
    sequences = collect_sequences(record_name, chunk_size)

    if not sequences:
        raise ValueError(f"Record {record_name} emitted no streaming sequences.")

    labels_by_peak = resolve_ground_truth(record_name)
    fp32_events, int8_events = classify_with_both(
        sequences,
        fp32_classifier,
        int8_classifier,
    )

    target_peaks = [event.target_peak_index for event in fp32_events]
    fp32_all = np.asarray(
        [event.predicted_class_index for event in fp32_events],
        dtype=np.int64,
    )
    int8_all = np.asarray(
        [event.predicted_class_index for event in int8_events],
        dtype=np.int64,
    )

    labelled_positions, true_indices = align_ground_truth(
        target_peaks,
        labels_by_peak,
    )

    observed = len(sequences)
    labelled = int(labelled_positions.size)
    excluded = observed - labelled

    if labelled == 0:
        raise ValueError(f"Record {record_name} has no labelled sequences.")

    changed_all = int(np.sum(fp32_all != int8_all))

    fp32_predictions = fp32_all[labelled_positions]
    int8_predictions = int8_all[labelled_positions]

    fp32_metrics = classification_metrics(true_indices, fp32_predictions)
    int8_metrics = classification_metrics(true_indices, int8_predictions)
    outcomes = changed_outcomes(true_indices, fp32_predictions, int8_predictions)

    record_result = {
        "record_name": record_name,
        "chunk_size": chunk_size,
        "streaming_sequences_observed": observed,
        "labelled_sequences_evaluated": labelled,
        "unlabelled_sequences_excluded": excluded,
        "changed_predictions_excluded_from_ground_truth": (
            changed_all - outcomes["num_changed"]
        ),
        "fp32": fp32_metrics,
        "int8": int8_metrics,
        "deltas": metric_deltas(fp32_metrics, int8_metrics),
        "changed_outcomes": outcomes,
    }

    logger.info(
        "Record %s: %d labelled of %d observed | FP32 acc %.4f | INT8 acc "
        "%.4f | changed %d (harmful %d, helpful %d, both wrong %d)",
        record_name,
        labelled,
        observed,
        fp32_metrics["accuracy"],
        int8_metrics["accuracy"],
        outcomes["num_changed"],
        outcomes["fp32_correct_int8_wrong"],
        outcomes["fp32_wrong_int8_correct"],
        outcomes["both_wrong"],
    )

    pooled = {
        "true": true_indices,
        "fp32": fp32_predictions,
        "int8": int8_predictions,
    }

    return record_result, pooled


# ---------------------------------------------------------------------
#                              Aggregate
# ---------------------------------------------------------------------


def build_aggregate(
    record_results: list[dict],
    pooled: dict[str, NDArray[np.int64]],
    failed_records: list[str] | None = None,
) -> dict:
    """
    Aggregate over the pooled labelled arrays of every record.

    Metrics are recomputed from concatenated predictions so each record
    contributes in proportion to its labelled sequence count.
    """

    failed_records = list(failed_records or [])

    fp32_metrics = classification_metrics(pooled["true"], pooled["fp32"])
    int8_metrics = classification_metrics(pooled["true"], pooled["int8"])
    outcomes = changed_outcomes(pooled["true"], pooled["fp32"], pooled["int8"])

    return {
        "num_records_evaluated": len(record_results),
        "record_names": [result["record_name"] for result in record_results],
        "total_streaming_sequences_observed": sum(
            int(result["streaming_sequences_observed"]) for result in record_results
        ),
        "total_labelled_sequences_evaluated": int(pooled["true"].size),
        "total_unlabelled_sequences_excluded": sum(
            int(result["unlabelled_sequences_excluded"]) for result in record_results
        ),
        "changed_predictions_excluded_from_ground_truth": sum(
            int(result["changed_predictions_excluded_from_ground_truth"])
            for result in record_results
        ),
        "fp32": fp32_metrics,
        "int8": int8_metrics,
        "deltas": metric_deltas(fp32_metrics, int8_metrics),
        "changed_outcomes": outcomes,
        "per_record": record_results,
        "failed_records": failed_records,
    }


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score FP32 and INT8 predictions against the same supported "
            "ground-truth labels on identical streaming sequences."
        )
    )
    parser.add_argument(
        "--fp32-model-path",
        type=Path,
        default=DEFAULT_FP32_MODEL_PATH,
    )
    parser.add_argument(
        "--int8-model-path",
        type=Path,
        default=DEFAULT_INT8_MODEL_PATH,
    )
    parser.add_argument("--record-name", type=str, default=DEFAULT_RECORD_NAME)
    parser.add_argument(
        "--all-validation-records",
        action="store_true",
        help="Evaluate every record in the validation split.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--write-plots",
        action="store_true",
        help="Save confusion matrices and metric comparison figures.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Extract CL arguments
    args = parse_args()

    if args.all_validation_records:
        record_names = load_split_record_names(args.split_summary_path, "val")
    else:
        record_names = [args.record_name]

    # Setup the fp32 and int8 models
    fp32_classifier = ONNXSequenceClassifier(args.fp32_model_path)
    int8_classifier = ONNXSequenceClassifier(args.int8_model_path)

    record_results: list[dict] = []
    # This will collect one numpy array per record.
    pooled_parts: dict[str, list[NDArray[np.int64]]] = {
        "true": [],
        "fp32": [],
        "int8": [],
    }
    failed_records: list[str] = []

    # For each record
    for record_name in record_names:
        try:
            record_result, pooled = evaluate_record(
                record_name=record_name,
                fp32_classifier=fp32_classifier,
                int8_classifier=int8_classifier,
                chunk_size=args.chunk_size,
            )
        except Exception:
            failed_records.append(record_name)
            logger.exception("Record %s could not be evaluated", record_name)
            continue

        record_results.append(record_result)

        for key in pooled_parts:
            pooled_parts[key].append(pooled[key])

    if not record_results:
        logger.error("No record could be evaluated.")
        raise SystemExit(1)

    pooled = {key: np.concatenate(parts) for key, parts in pooled_parts.items()}
    aggregate = build_aggregate(record_results, pooled, failed_records)

    _write_json(aggregate, args.output_dir / SUMMARY_FILENAME)

    np.savez_compressed(
        args.output_dir / ARRAYS_FILENAME,
        true_indices=pooled["true"],
        fp32_predictions=pooled["fp32"],
        int8_predictions=pooled["int8"],
    )

    if args.write_plots:
        from ecg_arrhythmia.evaluation.quantized_model_performance_plots import (
            write_performance_figures,
        )

        written = write_performance_figures(
            fp32_metrics=aggregate["fp32"],
            int8_metrics=aggregate["int8"],
            deltas=aggregate["deltas"],
            outcomes=aggregate["changed_outcomes"],
            figures_dir=args.figures_dir,
        )

        for path in written:
            logger.info("Wrote figure %s", path)

    logger.info("Wrote summary to %s", args.output_dir / SUMMARY_FILENAME)

    if failed_records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
