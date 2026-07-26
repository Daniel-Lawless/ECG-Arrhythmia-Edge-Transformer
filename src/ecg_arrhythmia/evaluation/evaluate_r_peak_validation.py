import json
from dataclasses import asdict
from pathlib import Path

from ecg_arrhythmia.detection.elgendi_detector import ElgendiDetector
from ecg_arrhythmia.detection.hamilton_detector import HamiltonDetector
from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.detection.xqrs_detector import XQRSDetector
from ecg_arrhythmia.evaluation.r_peak_evaluator import (
    DatasetRPeakEvaluation,
    evaluate_r_peak_records,
)

SPLIT_SUMMARY_PATH = Path("data/splits_sequences_matched/split_summary_metrics.json")

OUTPUT_DIR = Path("artifacts/results/detection_evaluation")

SPLIT_NAME = "validation"

# Matching tolerance for the main comparison. Converted to samples
# separately for each record using that record's sampling rate.
MATCHING_TOLERANCE_MS = 100.0

# Expected raw validation records for the current split. This is used
# only as a sanity check; the authoritative record list is read from the
# split summary so the pipeline follows the split rather than a hard-coded
# constant.
EXPECTED_VALIDATION_RECORDS = ["114", "122", "209", "210", "231", "233"]


def load_validation_record_names(
    summary_path: Path,
) -> list[str]:
    """
    Load validation patient IDs and expand grouped patients such as
    201_202 into their individual raw MIT-BIH record names.
    """

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    validation_patient_ids = summary["per_split"]["val"]["selected_patient_ids"]

    return [
        record_name
        for patient_id in validation_patient_ids
        for record_name in patient_id.split("_")
    ]


def build_detectors() -> list[RPeakDetector]:
    """
    Create the R-peak detectors to compare.

    The evaluator treats every entry as a generic ``RPeakDetector``, so
    detectors can be added or removed here without touching the
    evaluation code.
    """

    return [
        XQRSDetector(learn=True),
        HamiltonDetector(),
        ElgendiDetector(),
    ]


def summarise_evaluation(
    evaluation: DatasetRPeakEvaluation,
) -> dict[str, object]:
    """Build one compact comparison row from a dataset evaluation."""

    metrics = evaluation.metrics

    return {
        "detector": evaluation.detector_name,
        "records": metrics.num_records,
        "annotations": metrics.num_annotations,
        "detections": metrics.num_detections,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "mean_absolute_offset_ms": metrics.mean_absolute_offset_ms,
        "maximum_absolute_offset_ms": metrics.maximum_absolute_offset_ms,
        "runtime_seconds": evaluation.total_detector_runtime_seconds,
        "processing_speedup": evaluation.processing_speedup,
    }


def _format_optional(
    value: float | None,
    width: int,
    decimals: int,
) -> str:
    """Right-align an optional metric, showing "n/a" when it is undefined."""

    if value is None:
        return f"{'n/a':>{width}}"

    return f"{value:>{width}.{decimals}f}"


def print_comparison_table(
    comparison_rows: list[dict[str, object]],
) -> None:
    """Print a readable detector-comparison table.

    Recall and false negatives are shown alongside F1 because missed
    beats are especially important for this medical-signal
    preprocessing stage.
    """

    header = (
        f"{'Detector':<10}"
        f"{'Recs':>5}"
        f"{'Annot':>7}"
        f"{'Detect':>8}"
        f"{'TP':>7}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'Precision':>11}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'MAE(ms)':>10}"
        f"{'MaxAE(ms)':>11}"
        f"{'Runtime(s)':>12}"
        f"{'Speed(x)':>10}"
    )

    print(header)
    print("-" * len(header))

    for row in comparison_rows:
        print(
            f"{row['detector']:<10}"
            f"{row['records']:>5}"
            f"{row['annotations']:>7}"
            f"{row['detections']:>8}"
            f"{row['true_positives']:>7}"
            f"{row['false_positives']:>6}"
            f"{row['false_negatives']:>6}"
            f"{row['precision']:>11.6f}"
            f"{row['recall']:>10.6f}"
            f"{row['f1']:>10.6f}"
            f"{_format_optional(row['mean_absolute_offset_ms'], 10, 3)}"
            f"{_format_optional(row['maximum_absolute_offset_ms'], 11, 3)}"
            f"{row['runtime_seconds']:>12.3f}"
            f"{row['processing_speedup']:>10.1f}"
        )


def print_symbol_recall_table(
    evaluations: list[DatasetRPeakEvaluation],
) -> None:
    """Print per-symbol recall for every detector.

    Strong overall recall can hide poor detection of unusual beat
    morphologies, so recall is broken down by the original MIT-BIH
    annotation symbol.
    """

    # Collect the union of symbols across all detectors so the table has
    # one consistent row per symbol.
    symbols: set[str] = set()
    for evaluation in evaluations:
        symbols.update(evaluation.symbol_metrics)

    header = f"{'Symbol':<8}{'Annot':>8}"
    for evaluation in evaluations:
        header += f"{evaluation.detector_name:>12}"

    print(header)
    print("-" * len(header))

    for symbol in sorted(symbols):
        # Annotation totals are identical across detectors, so take the
        # count from whichever detector observed this symbol.
        annotation_total = 0
        for evaluation in evaluations:
            symbol_metrics = evaluation.symbol_metrics.get(symbol)
            if symbol_metrics is not None:
                annotation_total = symbol_metrics.annotations
                break

        row = f"{symbol:<8}{annotation_total:>8}"
        for evaluation in evaluations:
            symbol_metrics = evaluation.symbol_metrics.get(symbol)
            recall = symbol_metrics.recall if symbol_metrics is not None else 0.0
            row += f"{recall:>12.4f}"

        print(row)


def main() -> None:
    # Load the record names from our matched split
    record_names = load_validation_record_names(SPLIT_SUMMARY_PATH)

    print(f"Validation records: {record_names}")

    # the extracted record names should be what we expect
    if record_names == EXPECTED_VALIDATION_RECORDS:
        print("Sanity check: validation records match the expected split.")
    else:
        print(
            "Sanity check WARNING: validation records "
            f"{record_names} differ from the expected "
            f"{EXPECTED_VALIDATION_RECORDS}."
        )

    # Create a list of each detector
    detectors = build_detectors()

    # Create our directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluations: list[DatasetRPeakEvaluation] = []
    comparison_rows: list[dict[str, object]] = []

    # For each detector
    for detector in detectors:
        print(f"\nEvaluating detector: {detector.name}")

        # Evaluates a detector on all records in our validation set
        # and returns a DatasetRPeakEvaluation object
        evaluation = evaluate_r_peak_records(
            detector=detector,
            record_names=record_names,
            tolerance_ms=MATCHING_TOLERANCE_MS,
            split_name=SPLIT_NAME,
        )

        # Add this evaluation object
        evaluations.append(evaluation)
        comparison_rows.append(summarise_evaluation(evaluation))

        # Save the detailed per-detector evaluation.
        detector_output_path = OUTPUT_DIR / f"{detector.name}_metrics.json"
        with detector_output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(evaluation), file, indent=2)

        print(f"Saved detailed results to {detector_output_path}")

    # Rank primarily by F1, then recall, so the strongest detector for
    # this preprocessing stage appears first. Runtime is reported but is
    # not used to declare a winner.
    comparison_rows.sort(
        key=lambda row: (row["f1"], row["recall"]),
        reverse=True,
    )

    comparison = {
        "split_name": SPLIT_NAME,
        "tolerance_ms": MATCHING_TOLERANCE_MS,
        "record_names": record_names,
        "detectors": comparison_rows,
    }

    comparison_output_path = OUTPUT_DIR / "detector_comparison.json"
    with comparison_output_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    print(
        f"\nMatching tolerance: {MATCHING_TOLERANCE_MS:.1f} ms "
        "(converted to samples per record)"
    )

    print("\nDetector comparison (ranked by F1, then recall):\n")
    print_comparison_table(comparison_rows)

    print("\nPer-symbol recall by detector:\n")
    print_symbol_recall_table(evaluations)

    print(f"\nSaved combined comparison to {comparison_output_path}")


if __name__ == "__main__":
    main()
