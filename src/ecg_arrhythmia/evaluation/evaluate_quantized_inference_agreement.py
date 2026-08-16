import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from ecg_arrhythmia.data.build_xqrs_centered_dataset import load_split_record_names
from ecg_arrhythmia.data.label_mapping import CLASS_LABELS, NUM_CLASSES
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import collect_sequences
from ecg_arrhythmia.streaming.onnx_sequence_classifier import (
    ONNXSequenceClassifier,
    PredictionEvent,
)
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/deployment_evaluation/quantization_agreement"
)
DEFAULT_FIGURES_DIR = Path("artifacts/figures/quantization_agreement")
DEFAULT_FP32_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_INT8_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer_int8.onnx")

DISAGREEMENTS_FILENAME = "fp32_vs_int8_disagreements.json"
SUMMARY_FILENAME = "quantization_agreement_summary.json"


# ---------------------------------------------------------------------
#                          Comparison Helpers
# ---------------------------------------------------------------------


def agreement_matrix(
    reference_classes: NDArray[np.integer],
    comparison_classes: NDArray[np.integer],
) -> list[list[int]]:
    """
    Count how often each reference class met each comparison class.

    Rows are reference predictions and columns are comparison
    predictions. This is agreement between two inference paths, not
    accuracy against ground truth, so full parity puts every count on the
    diagonal.
    """

    return confusion_matrix(
        reference_classes, comparison_classes, labels=range(NUM_CLASSES)
    ).tolist()


def logit_margins(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Winning logit minus second-highest logit, per sequence.

    A small margin means the model was already close to choosing another
    class.
    """

    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(
            f"Logits must have shape (num_sequences, num_classes), "
            f"found {logits.shape}."
        )

    # Sorts each sequence, and takes the final two elements
    # from each row sequence
    top_two = np.sort(logits, axis=1)[:, -2:]

    # Then, highest logit - second-highest logit
    return (top_two[:, 1] - top_two[:, 0]).astype(np.float32)


def transition_counts(matrix: list[list[int]]) -> dict[str, int]:
    """
    Every off-diagonal FP32-to-INT8 class transition, generated
    systematically so no direction is assumed in advance.
    """

    counts = np.asarray(matrix, dtype=np.int64)

    return {
        f"{CLASS_LABELS[row]}_to_{CLASS_LABELS[column]}": int(counts[row, column])
        for row in range(NUM_CLASSES)
        for column in range(NUM_CLASSES)
        if row != column
    }


def compare_record_arrays(
    fp32_logits: NDArray[np.float32],
    int8_logits: NDArray[np.float32],
    target_peaks: NDArray[np.int64],
) -> tuple[dict, dict]:
    """
    Compare both models' logits for one record's sequences.

    Returns a JSON-ready summary and the underlying arrays, so callers
    can persist or plot without recomputing.
    """

    # Convert to a numpy array of type float 32 if it is not already
    fp32 = np.asarray(fp32_logits, dtype=np.float32)
    int8 = np.asarray(int8_logits, dtype=np.float32)
    target_peaks = np.asarray(target_peaks, dtype=np.int64)

    if fp32.shape != int8.shape:
        raise ValueError(
            f"FP32 and INT8 logit shapes must match, found {fp32.shape} "
            f"and {int8.shape}."
        )

    if fp32.ndim != 2 or fp32.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Logits must have shape (num_sequences, {NUM_CLASSES}), "
            f"found {fp32.shape}."
        )

    if fp32.shape[0] == 0:
        raise ValueError("At least one sequence is required to compare.")

    if target_peaks.shape[0] != fp32.shape[0]:
        raise ValueError(
            f"One target peak per sequence is required: {target_peaks.shape[0]} "
            f"peaks for {fp32.shape[0]} sequences."
        )

    num_sequences = int(fp32.shape[0])
    # This calculates the element by element difference in logits.
    difference = np.abs(fp32 - int8)
    # This gives us the max difference in each sequence
    per_sequence_max = difference.max(axis=1)

    # Returns the index of the largest logit in each sequence
    fp32_classes = fp32.argmax(axis=1)
    int8_classes = int8.argmax(axis=1)
    # If those indices match, then they have predicted the same class.
    agreed = fp32_classes == int8_classes

    # Return the agreement matrix as a Python list.
    matrix = agreement_matrix(fp32_classes, int8_classes)

    # Calculate the logit margin for each sequence in the FP32 ONNX model
    fp32_margins = logit_margins(fp32)
    # Calculate the logit margin for each sequence in the INT8 Quantized model
    int8_margins = logit_margins(int8)

    # Flip the agreed boolean array to give a disagreemnts array and sum
    # across its true values.
    disagreements = int(np.sum(~agreed))

    summary = {
        "num_sequences_compared": num_sequences,
        "class_agreements": int(np.sum(agreed)),
        "class_disagreements": disagreements,
        "class_agreement_percentage": float(np.mean(agreed) * 100.0),
        "class_disagreement_percentage": float(np.mean(~agreed) * 100.0),
        "mean_absolute_logit_difference": float(difference.mean()),
        "maximum_absolute_logit_difference": float(difference.max()),
        "mean_per_sequence_maximum_absolute_logit_difference": float(
            per_sequence_max.mean()
        ),
        "median_per_sequence_maximum_absolute_logit_difference": float(
            np.percentile(per_sequence_max, 50)
        ),
        "p95_per_sequence_maximum_absolute_logit_difference": float(
            np.percentile(per_sequence_max, 95)
        ),
        "agreement_matrix": matrix,
        "transition_counts": transition_counts(matrix),
        "disagreement_target_peaks": [int(peak) for peak in target_peaks[~agreed]],
        "fp32_logit_margin_mean_agreeing": (
            float(fp32_margins[agreed].mean()) if agreed.any() else None
        ),
        "fp32_logit_margin_mean_disagreeing": (
            float(fp32_margins[~agreed].mean()) if disagreements else None
        ),
    }
    arrays = {
        "difference": difference,
        "per_sequence_max": per_sequence_max,
        "fp32_classes": fp32_classes,
        "int8_classes": int8_classes,
        "agreed": agreed,
        "fp32_margins": fp32_margins,
        "int8_margins": int8_margins,
    }

    return summary, arrays


def classify_with_both(
    sequences: list[BeatSequence],
    fp32_classifier: ONNXSequenceClassifier,
    int8_classifier: ONNXSequenceClassifier,
) -> tuple[list[PredictionEvent], list[PredictionEvent]]:
    """
    Run both classifiers over the identical sequence objects, in order.

    Both loops iterate the same list, so the two models can never see
    differently preprocessed inputs.
    """

    if not sequences:
        raise ValueError("At least one sequence is required to compare.")

    # This will give us two lists of PredictionEvent objects
    fp32_events = [fp32_classifier.predict(sequence) for sequence in sequences]
    int8_events = [int8_classifier.predict(sequence) for sequence in sequences]

    return fp32_events, int8_events


def disagreement_entries(
    record_name: str,
    fp32_events: list[PredictionEvent],
    int8_events: list[PredictionEvent],
    arrays: dict,
) -> list[dict]:
    """One compact, traceable entry per changed prediction."""

    entries = []

    # ~arrays["agreed"] flips the agreement booleans, so True marks disagreements.
    # np.flatnonzero() returns the indices of those True values.
    # Therefore, loop over the index of each disagreement:
    for index in np.flatnonzero(~arrays["agreed"]):
        # extract the PredictionEvent object of the disagrement for
        # both models
        fp32_event = fp32_events[index]
        int8_event = int8_events[index]

        # Append a summary dictionary to entries for each disagreement
        entries.append(
            {
                "record_name": record_name,
                "target_peak_index": fp32_event.target_peak_index,
                "peak_indices": list(fp32_event.peak_indices),
                "fp32_predicted_label": fp32_event.predicted_label,
                "int8_predicted_label": int8_event.predicted_label,
                "fp32_logits": [float(v) for v in fp32_event.logits],
                "int8_logits": [float(v) for v in int8_event.logits],
                "absolute_logit_differences": [
                    float(v) for v in arrays["difference"][index]
                ],
                "maximum_absolute_logit_difference": float(
                    arrays["per_sequence_max"][index]
                ),
                "fp32_logit_margin": float(arrays["fp32_margins"][index]),
                "int8_logit_margin": float(arrays["int8_margins"][index]),
            }
        )

    return entries


def _largest_drift_entry(
    fp32_events: list[PredictionEvent],
    int8_events: list[PredictionEvent],
    arrays: dict,
) -> dict:
    """The sequence with the largest per-sequence logit drift."""

    # Is the index of the sequence with the largest logit difference
    # between fp32 and int 8
    index = int(np.argmax(arrays["per_sequence_max"]))
    # Return the PredictionEvent object of this sequence
    fp32_event = fp32_events[index]
    int8_event = int8_events[index]

    return {
        "target_peak_index": fp32_event.target_peak_index,
        "fp32_predicted_label": fp32_event.predicted_label,
        "int8_predicted_label": int8_event.predicted_label,
        "fp32_logits": [float(v) for v in fp32_event.logits],
        "int8_logits": [float(v) for v in int8_event.logits],
        "maximum_absolute_logit_difference": float(arrays["per_sequence_max"][index]),
    }


# ---------------------------------------------------------------------
#                            Per-Record Entry
# ---------------------------------------------------------------------


def evaluate_record(
    record_name: str,
    fp32_classifier: ONNXSequenceClassifier,
    int8_classifier: ONNXSequenceClassifier,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> tuple[dict, NDArray[np.float32], list[dict]]:
    """
    Compare both models over one record's streaming sequences.

    Returns the record summary, the per-sequence maximum differences
    (for pooled aggregate statistics) and the disagreement entries, so
    the caller can release the record's full arrays afterwards.
    """

    # Collect all sequences for this record
    sequences = collect_sequences(record_name, chunk_size)

    if not sequences:
        raise ValueError(f"Record {record_name} emitted no streaming sequences.")

    # Return the predictions of each model
    fp32_events, int8_events = classify_with_both(
        sequences,
        fp32_classifier,
        int8_classifier,
    )

    # Extract the logits from each PredictionEvent object for each model
    # and stack them on top of each other
    fp32_logits = np.stack([event.logits for event in fp32_events])
    int8_logits = np.stack([event.logits for event in int8_events])

    # Extract the target_peaks indices from the ONNX FP32 model PredictionEvent object
    target_peaks = np.asarray(
        [event.target_peak_index for event in fp32_events],
        dtype=np.int64,
    )

    # returns a summary of the logits between fp32 and int8
    summary, arrays = compare_record_arrays(fp32_logits, int8_logits, target_peaks)

    record_result = {
        "record_name": record_name,
        "chunk_size": chunk_size,
        **summary,
        # The sequence where fp32 and int8 disagreed the most
        "largest_drift_sequence": _largest_drift_entry(
            fp32_events,
            int8_events,
            arrays,
        ),
    }
    # Returns a list of dictionaries, each one summarising each disagreement
    disagreements = disagreement_entries(
        record_name,
        fp32_events,
        int8_events,
        arrays,
    )

    logger.info(
        "Record %s: %d sequences, %.4f%% agreement, %d disagreements, max |drift| %.3e",
        record_name,
        summary["num_sequences_compared"],
        summary["class_agreement_percentage"],
        summary["class_disagreements"],
        summary["maximum_absolute_logit_difference"],
    )

    if output_dir is not None:
        _write_json(record_result, output_dir / f"record_{record_name}.json")

        np.savez_compressed(
            output_dir / f"record_{record_name}_logits.npz",
            fp32_logits=fp32_logits,
            int8_logits=int8_logits,
            target_peaks=target_peaks,
            fp32_classes=arrays["fp32_classes"],
            int8_classes=arrays["int8_classes"],
            per_sequence_maximum_difference=arrays["per_sequence_max"],
            fp32_margins=arrays["fp32_margins"],
            int8_margins=arrays["int8_margins"],
        )

    if figures_dir is not None:
        from ecg_arrhythmia.evaluation.quantization_agreement_plots import (
            write_record_agreement_figures,
        )

        written = write_record_agreement_figures(
            record_name=record_name,
            fp32_logits=fp32_logits,
            int8_logits=int8_logits,
            target_peaks=target_peaks,
            matrix=summary["agreement_matrix"],
            fp32_margins=arrays["fp32_margins"],
            agreed=arrays["agreed"],
            figures_dir=figures_dir,
        )
        for path in written:
            logger.info("Wrote figure %s", path)

    return record_result, arrays["per_sequence_max"], disagreements


# ---------------------------------------------------------------------
#                              Aggregate
# ---------------------------------------------------------------------


def build_aggregate(
    record_results: list[dict],
    per_sequence_max_arrays: list[NDArray[np.float32]],
    chunk_size: int,
    failed_records: list[str] | None = None,
) -> dict:
    """
    Pool per-record results into aggregate agreement and drift.

    Every ratio is computed from pooled counts and every percentile from
    pooled per-sequence values, so records with more sequences weigh
    proportionally more; record-level percentages are never averaged.
    """

    failed_records = list(failed_records or [])

    if len(record_results) != len(per_sequence_max_arrays):
        raise ValueError(
            "One per-sequence difference array is required per record result."
        )

    # Calculate total sequences, agreements, and disagreements for
    # all evaluated records
    total_sequences = sum(
        int(result["num_sequences_compared"]) for result in record_results
    )
    total_agreements = sum(int(result["class_agreements"]) for result in record_results)
    total_disagreements = sum(
        int(result["class_disagreements"]) for result in record_results
    )

    # Each sequence contributes the same number of logits, so weighting
    # the record means by sequence count recovers the exact pooled mean.
    weighted_mean = sum(
        float(result["mean_absolute_logit_difference"])
        * int(result["num_sequences_compared"])
        for result in record_results
    )

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for result in record_results:
        matrix += np.asarray(result["agreement_matrix"], dtype=np.int64)

    pooled_max = (
        np.concatenate(per_sequence_max_arrays)
        if per_sequence_max_arrays
        else np.zeros(0, dtype=np.float32)
    )

    records_with_disagreements = [
        result["record_name"]
        for result in record_results
        if result["class_disagreements"] > 0
    ]

    return {
        "num_records_evaluated": len(record_results),
        "record_names": [result["record_name"] for result in record_results],
        "chunk_size": chunk_size,
        "total_sequences_compared": total_sequences,
        "total_class_agreements": total_agreements,
        "total_class_disagreements": total_disagreements,
        "class_agreement_percentage": (
            total_agreements / total_sequences * 100.0 if total_sequences else 0.0
        ),
        "class_disagreement_percentage": (
            total_disagreements / total_sequences * 100.0 if total_sequences else 0.0
        ),
        "mean_absolute_logit_difference": (
            weighted_mean / total_sequences if total_sequences else 0.0
        ),
        "maximum_absolute_logit_difference": max(
            (
                float(result["maximum_absolute_logit_difference"])
                for result in record_results
            ),
            default=0.0,
        ),
        "mean_per_sequence_maximum_absolute_logit_difference": (
            float(pooled_max.mean()) if pooled_max.size else 0.0
        ),
        "median_per_sequence_maximum_absolute_logit_difference": (
            float(np.percentile(pooled_max, 50)) if pooled_max.size else 0.0
        ),
        "p95_per_sequence_maximum_absolute_logit_difference": (
            float(np.percentile(pooled_max, 95)) if pooled_max.size else 0.0
        ),
        "agreement_matrix": matrix.tolist(),
        "transition_counts": transition_counts(matrix.tolist()),
        "records_with_class_disagreements": records_with_disagreements,
        "number_of_records_with_class_disagreements": len(records_with_disagreements),
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
            "Measure FP32 versus INT8 class agreement and logit drift on "
            "identical streaming-emitted beat sequences."
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
        help="Save agreement matrices and drift figures.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # If all validation record were passed, create a list of all validation
    # record names, else use the explicity passed record name
    if args.all_validation_records:
        record_names = load_split_record_names(args.split_summary_path, "val")
    else:
        record_names = [args.record_name]

    figures_dir = args.figures_dir if args.write_plots else None

    # Load the ONNX FP32 model and the INT8 quantised model
    fp32_classifier = ONNXSequenceClassifier(args.fp32_model_path)
    int8_classifier = ONNXSequenceClassifier(args.int8_model_path)

    record_results: list[dict] = []
    per_sequence_max_arrays: list[NDArray[np.float32]] = []
    all_disagreements: list[dict] = []
    failed_records: list[str] = []

    # For each record passed
    for record_name in record_names:
        try:
            # Returns a summary of this records results, the maximum logit difference
            # between fp32 and int8 for each sequence, and a summary of
            # the disagreements
            record_result, per_sequence_max, disagreements = evaluate_record(
                record_name=record_name,
                fp32_classifier=fp32_classifier,
                int8_classifier=int8_classifier,
                chunk_size=args.chunk_size,
                output_dir=args.output_dir,
                figures_dir=figures_dir,
            )
        except Exception:
            failed_records.append(record_name)
            logger.exception("Record %s could not be evaluated", record_name)
            continue

        # append this records result, per sequence max and disagreements to
        # our running totals
        record_results.append(record_result)
        per_sequence_max_arrays.append(per_sequence_max)
        all_disagreements.extend(disagreements)

    aggregate = build_aggregate(
        record_results,
        per_sequence_max_arrays,
        chunk_size=args.chunk_size,
        failed_records=failed_records,
    )

    _write_json(
        {
            "num_disagreements": len(all_disagreements),
            "disagreements": all_disagreements,
        },
        args.output_dir / DISAGREEMENTS_FILENAME,
    )
    _write_json(aggregate, args.output_dir / SUMMARY_FILENAME)

    if figures_dir is not None and record_results:
        from ecg_arrhythmia.evaluation.quantization_agreement_plots import (
            write_aggregate_agreement_figures,
        )

        written = write_aggregate_agreement_figures(
            matrix=aggregate["agreement_matrix"],
            pooled_per_sequence_max=np.concatenate(per_sequence_max_arrays),
            figures_dir=figures_dir,
        )
        for path in written:
            logger.info("Wrote figure %s", path)

    logger.info(
        "Wrote agreement summary to %s",
        args.output_dir / SUMMARY_FILENAME,
    )

    # Only records that could not be evaluated fail.
    if failed_records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
