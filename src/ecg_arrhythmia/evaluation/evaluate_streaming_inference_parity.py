import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ecg_arrhythmia.data.build_xqrs_centered_dataset import load_split_record_names
from ecg_arrhythmia.data.label_mapping import NUM_CLASSES
from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.deployment.verify_onnx_parity import load_pytorch_model
from ecg_arrhythmia.streaming.onnx_sequence_classifier import (
    ONNXSequenceClassifier,
    PredictionEvent,
)
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplaySource
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.streaming.streaming_predictor import StreamingPredictor

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/deployment_evaluation/streaming_inference_parity"
)
DEFAULT_FIGURES_DIR = Path("artifacts/figures/streaming_inference_parity")
DEFAULT_CHECKPOINT_PATH = Path("artifacts/models/ecg_sequence_transformer_tuned.pt")
DEFAULT_ONNX_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")

# Tuned three-layer transformer configuration behind the checkpoint.
NUM_LAYERS = 3
DROPOUT = 0.2

RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-5

# The three comparisons, as reference-versus-comparison logit sources.
COMPARISON_PAIRS = (
    ("pytorch_vs_offline_onnx", "pytorch", "offline_onnx"),
    ("offline_onnx_vs_streaming_onnx", "offline_onnx", "streaming_onnx"),
    ("pytorch_vs_streaming_onnx", "pytorch", "streaming_onnx"),
)


# ---------------------------------------------------------------------
#                        Collecting Identical Inputs
# ---------------------------------------------------------------------


class _RecordingClassifier(ONNXSequenceClassifier):
    """
    Keeps each sequence beside the prediction the live stream made for it.

    Recording happens around the real predict call, so the streaming
    result is genuinely the one the pipeline produced, and the retained
    sequence is the identical object the other two paths will consume.
    """

    def __init__(self, onnx_model_path: Path) -> None:
        super().__init__(onnx_model_path)
        self.sequences: list[BeatSequence] = []
        self.events: list[PredictionEvent] = []

    def predict(self, sequence: BeatSequence) -> PredictionEvent:
        event = super().predict(sequence)

        self.sequences.append(sequence)
        self.events.append(event)

        return event


def _stream_record(
    record_name: str,
    chunk_size: int,
    onnx_model_path: Path,
) -> tuple[list[BeatSequence], list[PredictionEvent]]:
    """Replay one record and capture every sequence and its prediction."""

    signals, fields, _ = load_record(record_name=record_name)
    signal, _ = select_signal_channel(signals=signals, fields=fields)

    source = ReplaySource(
        signal=signal,
        sampling_rate=float(fields["fs"]),
        chunk_size=chunk_size,
        record_name=record_name,
    )
    classifier = _RecordingClassifier(onnx_model_path)
    predictor = StreamingPredictor(
        engine=StreamingEngine(),
        classifier=classifier,
    )
    predictor.start_record(record_name=record_name)

    for chunk in source.iter_chunks():
        predictor.process_chunk(chunk)

    predictor.flush()

    logger.info(
        "Record %s produced %d streaming sequences",
        record_name,
        len(classifier.sequences),
    )

    return classifier.sequences, classifier.events


def _pytorch_logits(
    model: torch.nn.Module,
    sequences: list[BeatSequence],
) -> NDArray[np.float32]:
    """
    Run the tuned PyTorch model over the captured sequences.

    One sequence per forward pass, so PyTorch receives exactly the batch
    dimension and float32 values ONNX Runtime received.
    """

    rows: list[NDArray[np.float32]] = []

    model.eval()

    with torch.no_grad():
        for sequence in sequences:
            ecg = torch.tensor(sequence.ecg, dtype=torch.float32).unsqueeze(0)
            rr = torch.tensor(sequence.rr, dtype=torch.float32).unsqueeze(0)
            rows.append(model(ecg, rr).numpy()[0])

    return _stack(rows)


def _event_logits(events: list[PredictionEvent]) -> NDArray[np.float32]:
    return _stack([event.logits for event in events])


def _stack(rows: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    if not rows:
        return np.zeros((0, NUM_CLASSES), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------------
#                              Comparison
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

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (reference_classes, comparison_classes), 1)

    return matrix.tolist()


def compare_logits(
    reference: NDArray[np.float32],
    comparison: NDArray[np.float32],
    target_peaks: NDArray[np.integer],
    relative_tolerance: float = RELATIVE_TOLERANCE,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
) -> dict:
    """Compare two sets of logits for the same sequences."""

    reference = np.asarray(reference, dtype=np.float32)
    comparison = np.asarray(comparison, dtype=np.float32)
    target_peaks = np.asarray(target_peaks, dtype=np.int64)

    if reference.shape != comparison.shape:
        raise ValueError(
            f"Logit shapes must match, found {reference.shape} and {comparison.shape}."
        )

    num_sequences = int(reference.shape[0])
    difference = np.abs(reference - comparison)

    reference_classes = (
        reference.argmax(axis=1) if num_sequences else np.zeros(0, dtype=np.int64)
    )
    comparison_classes = (
        comparison.argmax(axis=1) if num_sequences else np.zeros(0, dtype=np.int64)
    )
    agreed = reference_classes == comparison_classes

    # Tolerance is judged per sequence so a failure can be traced back to
    # the beat that produced it, never averaged away across the record.
    within_tolerance = np.array(
        [
            np.allclose(
                reference_row,
                comparison_row,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            for reference_row, comparison_row in zip(
                reference,
                comparison,
                strict=True,
            )
        ],
        dtype=bool,
    )

    return {
        "num_sequences_compared": num_sequences,
        "num_class_agreements": int(np.sum(agreed)),
        "class_agreement_percentage": (
            float(np.mean(agreed) * 100.0) if num_sequences else 0.0
        ),
        "mean_absolute_logit_difference": (
            float(np.mean(difference)) if num_sequences else 0.0
        ),
        "maximum_absolute_logit_difference": (
            float(np.max(difference)) if num_sequences else 0.0
        ),
        "num_arrays_within_tolerance": int(np.sum(within_tolerance)),
        "num_arrays_outside_tolerance": int(np.sum(~within_tolerance)),
        "arrays_exactly_equal": bool(np.array_equal(reference, comparison)),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "class_disagreement_target_peaks": [
            int(peak) for peak in target_peaks[~agreed]
        ],
        "tolerance_failure_target_peaks": [
            int(peak) for peak in target_peaks[~within_tolerance]
        ],
        "agreement_matrix": agreement_matrix(
            reference_classes,
            comparison_classes,
        ),
        "passed": bool(num_sequences > 0 and agreed.all() and within_tolerance.all()),
    }


# ---------------------------------------------------------------------
#                            Per-Record Entry
# ---------------------------------------------------------------------


def evaluate_record(
    record_name: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    onnx_model_path: Path = DEFAULT_ONNX_MODEL_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict:
    """Compare all three inference paths for one record."""

    sequences, streaming_events = _stream_record(
        record_name=record_name,
        chunk_size=chunk_size,
        onnx_model_path=onnx_model_path,
    )

    if not sequences:
        raise ValueError(f"Record {record_name} emitted no streaming sequences.")

    model = load_pytorch_model(
        checkpoint_path=checkpoint_path,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    # A second, independently constructed session gives the offline path,
    # using the identical single-sequence contract the stream uses.
    offline_classifier = ONNXSequenceClassifier(onnx_model_path)
    offline_events = [offline_classifier.predict(sequence) for sequence in sequences]

    logits = {
        "pytorch": _pytorch_logits(model, sequences),
        "offline_onnx": _event_logits(offline_events),
        "streaming_onnx": _event_logits(streaming_events),
    }
    target_peaks = np.asarray(
        [event.target_peak_index for event in streaming_events],
        dtype=np.int64,
    )

    comparisons = {
        name: compare_logits(
            reference=logits[reference],
            comparison=logits[comparison],
            target_peaks=target_peaks,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        for name, reference, comparison in COMPARISON_PAIRS
    }
    parity_passed = all(comparison["passed"] for comparison in comparisons.values())

    logger.info(
        "Record %s parity %s | max |diff| PyTorch vs streaming ONNX: %.3e",
        record_name,
        "passed" if parity_passed else "FAILED",
        comparisons["pytorch_vs_streaming_onnx"]["maximum_absolute_logit_difference"],
    )

    if not parity_passed:
        _log_failures(record_name, comparisons)

    if output_dir is not None:
        _write_record_logits(
            record_name=record_name,
            logits=logits,
            target_peaks=target_peaks,
            output_dir=output_dir,
        )

    if figures_dir is not None:
        _write_record_plots(
            record_name=record_name,
            logits=logits,
            target_peaks=target_peaks,
            comparisons=comparisons,
            figures_dir=figures_dir,
        )

    return {
        "record_name": record_name,
        "chunk_size": chunk_size,
        "num_sequences_compared": len(sequences),
        "first_target_peak": int(target_peaks[0]),
        "last_target_peak": int(target_peaks[-1]),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "comparisons": comparisons,
        "parity_passed": parity_passed,
    }


def _log_failures(record_name: str, comparisons: dict) -> None:
    for name, comparison in comparisons.items():
        if comparison["passed"]:
            continue

        logger.error(
            "Record %s %s failed: %d class disagreements, %d arrays outside "
            "tolerance, max |diff| %.3e",
            record_name,
            name,
            len(comparison["class_disagreement_target_peaks"]),
            comparison["num_arrays_outside_tolerance"],
            comparison["maximum_absolute_logit_difference"],
        )


def _write_record_logits(
    record_name: str,
    logits: dict,
    target_peaks: NDArray[np.int64],
    output_dir: Path,
) -> None:
    """
    Save the raw logits behind this record's summary.

    These are result data rather than a figure, so they sit beside the
    per-record JSON. Keeping them out of the JSON is what lets the
    summary stay small while the arrays remain available.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    logits_path = output_dir / f"record_{record_name}_logits.npz"

    np.savez_compressed(
        logits_path,
        pytorch_logits=logits["pytorch"],
        offline_onnx_logits=logits["offline_onnx"],
        streaming_onnx_logits=logits["streaming_onnx"],
        target_peaks=target_peaks,
    )

    logger.info("Wrote logits to %s", logits_path)


def _write_record_plots(
    record_name: str,
    logits: dict,
    target_peaks: NDArray[np.int64],
    comparisons: dict,
    figures_dir: Path,
) -> None:
    """Save the parity figures. Only PNG files are written here."""

    from ecg_arrhythmia.evaluation.streaming_inference_plots import (
        write_record_figures,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)

    written = write_record_figures(
        record_name=record_name,
        pytorch_logits=logits["pytorch"],
        streaming_onnx_logits=logits["streaming_onnx"],
        target_peaks=target_peaks,
        comparisons=comparisons,
        figures_dir=figures_dir,
    )

    for path in written:
        logger.info("Wrote figure %s", path)


# ---------------------------------------------------------------------
#                              Aggregate
# ---------------------------------------------------------------------


def aggregate_records(
    record_results: list[dict],
    failed_records: list[str] | None = None,
) -> dict:
    """
    Combine per-record parity summaries.

    Pure over the per-record dictionaries so the verdict logic can be
    tested with synthetic summaries and never has to load a model.
    """

    failed_records = list(failed_records or [])
    total_sequences = sum(
        int(result["num_sequences_compared"]) for result in record_results
    )

    comparisons: dict[str, dict] = {}

    for name, _, _ in COMPARISON_PAIRS:
        per_record = [result["comparisons"][name] for result in record_results]
        compared = sum(int(item["num_sequences_compared"]) for item in per_record)

        matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for item in per_record:
            matrix += np.asarray(item["agreement_matrix"], dtype=np.int64)

        agreements = sum(int(item["num_class_agreements"]) for item in per_record)

        # Each record's mean covers the same number of logits per
        # sequence, so weighting by sequence count recovers the exact
        # overall mean.
        weighted_difference = sum(
            float(item["mean_absolute_logit_difference"])
            * int(item["num_sequences_compared"])
            for item in per_record
        )

        comparisons[name] = {
            "total_sequences_compared": compared,
            "total_class_agreements": agreements,
            "class_agreement_percentage": (
                float(agreements / compared * 100.0) if compared else 0.0
            ),
            "mean_absolute_logit_difference": (
                float(weighted_difference / compared) if compared else 0.0
            ),
            "maximum_absolute_logit_difference": max(
                (
                    float(item["maximum_absolute_logit_difference"])
                    for item in per_record
                ),
                default=0.0,
            ),
            "total_arrays_within_tolerance": sum(
                int(item["num_arrays_within_tolerance"]) for item in per_record
            ),
            "total_arrays_outside_tolerance": sum(
                int(item["num_arrays_outside_tolerance"]) for item in per_record
            ),
            "all_arrays_exactly_equal": all(
                bool(item["arrays_exactly_equal"]) for item in per_record
            ),
            "agreement_matrix": matrix.tolist(),
            "passed": bool(per_record) and all(item["passed"] for item in per_record),
        }

    records_failing_parity = [
        result["record_name"]
        for result in record_results
        if not result["parity_passed"]
    ]

    return {
        "num_records_evaluated": len(record_results),
        "record_names": [result["record_name"] for result in record_results],
        "chunk_size": record_results[0]["chunk_size"] if record_results else None,
        "total_sequences_compared": total_sequences,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "comparisons": comparisons,
        "failed_records": failed_records,
        "records_failing_parity": records_failing_parity,
        "all_records_parity_passed": bool(record_results)
        and not failed_records
        and not records_failing_parity,
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
            "Compare PyTorch, offline ONNX and streaming ONNX predictions "
            "over identical streaming-emitted beat sequences."
        )
    )
    parser.add_argument("--record-name", type=str, default=DEFAULT_RECORD_NAME)
    parser.add_argument(
        "--all-validation-records",
        action="store_true",
        help="Evaluate every record in the validation split.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--onnx-model-path",
        type=Path,
        default=DEFAULT_ONNX_MODEL_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
    )
    parser.add_argument(
        "--write-plots",
        action="store_true",
        help="Save the agreement matrices and numerical parity figures.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    if args.all_validation_records:
        record_names = load_split_record_names(args.split_summary_path, "val")
    else:
        record_names = [args.record_name]

    figures_dir = args.figures_dir if args.write_plots else None

    record_results: list[dict] = []
    failed_records: list[str] = []

    for record_name in record_names:
        try:
            result = evaluate_record(
                record_name=record_name,
                checkpoint_path=args.checkpoint_path,
                onnx_model_path=args.onnx_model_path,
                chunk_size=args.chunk_size,
                output_dir=args.output_dir,
                figures_dir=figures_dir,
            )
        except Exception:
            failed_records.append(record_name)
            logger.exception("Record %s could not be evaluated", record_name)
            continue

        record_results.append(result)
        _write_json(result, args.output_dir / f"record_{record_name}.json")

    aggregate = aggregate_records(record_results, failed_records)
    summary_path = args.output_dir / "streaming_inference_parity_summary.json"
    _write_json(aggregate, summary_path)

    logger.info("Wrote parity summary to %s", summary_path)

    if not aggregate["all_records_parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
