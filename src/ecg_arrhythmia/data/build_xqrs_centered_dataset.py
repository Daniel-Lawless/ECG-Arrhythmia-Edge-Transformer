from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.data.build_dataset import EXCLUDED_AAMI_LABELS, get_patient_id
from ecg_arrhythmia.data.label_mapping import map_labels_to_aami
from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.data.sequence_dataset import (
    SequenceSegment,
    create_record_sequences,
)
from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.detection.xqrs_detector import XQRSDetector
from ecg_arrhythmia.evaluation.r_peak_matching import match_r_peaks
from ecg_arrhythmia.preprocessing.beat_extraction import (
    BEAT_SYMBOLS,
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    extract_beats,
)

logger = logging.getLogger(__name__)

# Default sequence length used by the trained transformer.
SEQUENCE_LENGTH = 5

# Default matching tolerance for transferring expert labels to detections.
MATCHING_TOLERANCE_MS = 100.0

# Placeholder label for beats that are only ever used as sequence context.
# create_record_sequences only reads the label of the final (target) beat,
# and every context-only beat is filtered out before saving, so this value
# never reaches the saved dataset. It only needs to be a valid class string.
CONTEXT_PLACEHOLDER_LABEL = "N"

# Splits this builder can construct.
ALLOWED_SPLIT_NAMES = ("train", "val", "test")

# AAMI classes that may appear as scored targets.
SUPPORTED_AAMI_CLASSES = {"N", "S", "V", "F"}


def load_split_record_names(summary_path: Path, split_name: str) -> list[str]:
    """
    Load the raw MIT-BIH record names for one split from the split summary.

    Record names come from
    ``summary["per_split"][split_name]["selected_patient_ids"]`` and grouped
    patient IDs such as ``"201_202"`` are expanded into ``"201"`` and
    ``"202"``. Only ``"train"`` and ``"val"`` are supported; any other name
    (including ``"test"``) raises a clear error.
    """

    if split_name not in ALLOWED_SPLIT_NAMES:
        raise ValueError(
            f"Unsupported split name {split_name!r}. "
            f"Choose one of {list(ALLOWED_SPLIT_NAMES)}."
        )

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    per_split = summary["per_split"]
    if split_name not in per_split:
        raise ValueError(f"Split {split_name!r} was not found in {summary_path}.")

    patient_ids = per_split[split_name]["selected_patient_ids"]
    return [
        record_name
        for patient_id in patient_ids
        for record_name in patient_id.split("_")
    ]


def assert_splits_pairwise_disjoint(summary_path: Path) -> dict[str, list[str]]:
    """
    Confirm the train, validation, and test record sets do not overlap.

    Returns the resolved record names for each split so callers can reuse
    them. Raises a clear error if any pair of splits shares a record.
    """

    split_records = {
        name: load_split_record_names(summary_path, name)
        for name in ALLOWED_SPLIT_NAMES
    }

    names = list(split_records)
    for first_index, first_name in enumerate(names):
        first_set = set(split_records[first_name])
        for second_name in names[first_index + 1 :]:
            overlap = sorted(first_set & set(split_records[second_name]))
            if overlap:
                raise ValueError(
                    f"{first_name} and {second_name} records overlap: {overlap}."
                )

    return split_records


# ---------------------------------------------------------------------
#                          Data Our Types
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RecordConversion:
    """
    Detected-centred beats and per-record conversion statistics for one
    MIT-BIH validation record.

    Every emitted beat corresponds to one valid XQRS detection (a
    detection that is not the first of the record and whose window lies
    fully inside the signal). Beats are stored in chronological order.
    """

    record_id: str
    patient_id: str
    lead_name: str
    sampling_rate: float

    # Beat-level arrays, all of length num_valid_beats.
    windows: NDArray[np.float64]
    rr_features: NDArray[np.float64]
    labels_for_sequencing: NDArray[np.str_]
    is_target: NDArray[np.bool_]
    is_matched: NDArray[np.bool_]
    detected_samples: NDArray[np.int64]
    annotation_samples: NDArray[np.int64]
    offset_samples: NDArray[np.int64]
    offset_ms: NDArray[np.float64]
    symbols: NDArray[np.str_]
    aami_labels: NDArray[np.str_]

    # Per-record conversion counts.
    num_annotations: int
    num_detections: int
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_classifiable: int
    unsupported_removed: int
    boundary_removed: int
    insufficient_rr_removed: int

    @property
    def num_valid_beats(self) -> int:
        return int(self.windows.shape[0])


@dataclass(frozen=True)
class XqrsCenteredDataset:
    """Complete XQRS-centred sequence dataset for a split."""

    X_sequences: NDArray[np.float64]
    rr_sequences: NDArray[np.float64]
    y_labels: NDArray[np.str_]
    patient_ids: NDArray[np.str_]
    target_indices: NDArray[np.int64]

    # Audit arrays aligned row-for-row with the saved sequences.
    audit_records: NDArray[np.str_]
    audit_detected_samples: NDArray[np.int64]
    audit_annotation_samples: NDArray[np.int64]
    audit_offset_samples: NDArray[np.int64]
    audit_offset_ms: NDArray[np.float64]
    audit_symbols: NDArray[np.str_]
    audit_has_unmatched_context: NDArray[np.bool_]
    audit_num_unmatched_context: NDArray[np.int64]

    summary: dict[str, object]


# ---------------------------------------------------------------------
#                     Per-Record Detected-Centred Beats
# ---------------------------------------------------------------------


def build_record_detected_beats(
    record_name: str,
    detector: RPeakDetector,
    tolerance_ms: float,
    normalise_beats: bool,
    excluded_labels: set[str],
) -> RecordConversion:
    """
    Run one detector on a complete validation record and build
    detected-centred beats with transferred expert labels.

    RR features are computed over the full chronological detection
    timeline (including false positives), so every detection influences
    the RR context of subsequent detections. Only matched, classifiable
    detections become scored target beats; all other valid detections
    remain available as sequence context.
    """

    signals, fields, annotation = load_record(record_name)
    signal, lead_name = select_signal_channel(signals=signals, fields=fields)
    sampling_rate = float(fields["fs"])
    signal_length = len(signal)

    if annotation.symbol is None:
        raise ValueError(f"Record {record_name} contains no annotation symbols.")

    # Keep only expert annotations that represent genuine heartbeats.
    heartbeat_annotations = [
        (int(sample), symbol)
        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol,
            strict=True,
        )
        if symbol in BEAT_SYMBOLS
    ]

    annotation_samples = np.asarray(
        [sample for sample, _ in heartbeat_annotations],
        dtype=np.int64,
    )
    annotation_symbols = np.asarray(
        [symbol for _, symbol in heartbeat_annotations],
        dtype=str,
    )

    # Run the detector once on the continuous raw ECG.
    detected_samples = detector.detect(signal=signal, sampling_rate=sampling_rate)

    if detected_samples.size == 0:
        raise ValueError(f"Detector produced no detections for record {record_name}.")

    # One-to-one matching between expert heartbeats and detections.
    tolerance_samples = round(tolerance_ms * sampling_rate / 1000.0)
    match_result = match_r_peaks(
        annotation_samples=annotation_samples,
        detected_samples=detected_samples,
        tolerance_samples=tolerance_samples,
    )

    # Map each matched detection to its transferred expert symbol / class.
    matched_detection_symbol: dict[int, str] = {}
    matched_detection_annotation: dict[int, int] = {}

    matched_symbols = annotation_symbols[match_result.matched_annotation_indices]
    matched_aami = (
        map_labels_to_aami(matched_symbols)
        if matched_symbols.size > 0
        else np.empty(0, dtype=str)
    )

    for pair_index in range(match_result.matched_detection_indices.size):
        detection_index = int(match_result.matched_detection_indices[pair_index])
        annotation_index = int(match_result.matched_annotation_indices[pair_index])
        matched_detection_symbol[detection_index] = str(
            annotation_symbols[annotation_index]
        )
        matched_detection_annotation[detection_index] = annotation_index

    matched_detection_class = {
        int(match_result.matched_detection_indices[pair_index]): str(
            matched_aami[pair_index]
        )
        for pair_index in range(match_result.matched_detection_indices.size)
    }

    # Extract detected-centred beats and RR features by reusing the tested
    # beat extractor. Every detection is treated as a heartbeat so that no
    # detection is dropped from the RR timeline. extract_beats skips the
    # first detection (no previous RR) and any detection whose window falls
    # outside the signal, exactly matching the expert pipeline.
    pseudo_symbols = ["N"] * detected_samples.size
    windows, _, rr_features = extract_beats(
        signal=signal,
        annotation_samples=detected_samples,
        annotation_symbols=pseudo_symbols,
        normalise=normalise_beats,
    )

    # Recreate the same emit decision so each emitted beat can be traced
    # back to its detection index. The first detection is the RR seed and
    # is always dropped; remaining detections are kept only when the full
    # window lies inside the signal.
    within_bounds = (detected_samples - SAMPLES_BEFORE >= 0) & (
        detected_samples + SAMPLES_AFTER <= signal_length
    )
    emit_mask = within_bounds.copy()
    emit_mask[0] = False
    emitted_detection_indices = np.nonzero(emit_mask)[0]

    if emitted_detection_indices.size != windows.shape[0]:
        raise ValueError(
            f"Record {record_name}: emitted-beat mismatch between "
            f"extract_beats ({windows.shape[0]}) and the reconstructed "
            f"detection mapping ({emitted_detection_indices.size})."
        )

    num_valid_beats = emitted_detection_indices.size

    # Build the per-beat audit and label arrays for the emitted beats.
    is_matched = np.zeros(num_valid_beats, dtype=bool)
    is_target = np.zeros(num_valid_beats, dtype=bool)
    annotation_sample_per_beat = np.full(num_valid_beats, -1, dtype=np.int64)
    offset_samples = np.zeros(num_valid_beats, dtype=np.int64)
    symbols = np.empty(num_valid_beats, dtype="<U2")
    aami_labels = np.empty(num_valid_beats, dtype="<U2")
    labels_for_sequencing = np.empty(num_valid_beats, dtype="<U2")
    detected_sample_per_beat = detected_samples[emitted_detection_indices].astype(
        np.int64
    )

    for beat_index, detection_index in enumerate(emitted_detection_indices.tolist()):
        detection_index = int(detection_index)
        symbols[beat_index] = ""
        aami_labels[beat_index] = ""
        labels_for_sequencing[beat_index] = CONTEXT_PLACEHOLDER_LABEL

        if detection_index not in matched_detection_symbol:
            # Unmatched detection: a false positive kept only as context.
            continue

        is_matched[beat_index] = True
        annotation_index = matched_detection_annotation[detection_index]
        annotation_sample = int(annotation_samples[annotation_index])
        annotation_sample_per_beat[beat_index] = annotation_sample
        offset_samples[beat_index] = (
            detected_sample_per_beat[beat_index] - annotation_sample
        )

        symbol = matched_detection_symbol[detection_index]
        aami_class = matched_detection_class[detection_index]
        symbols[beat_index] = symbol
        aami_labels[beat_index] = aami_class

        # A detection is a scored target only when its transferred class is
        # a supported class (Q and other excluded classes are context only).
        if aami_class not in excluded_labels:
            is_target[beat_index] = True
            labels_for_sequencing[beat_index] = aami_class

    offset_ms = offset_samples.astype(np.float64) * 1000.0 / sampling_rate

    # Per-record conversion statistics.
    true_positives = match_result.true_positives
    false_positives = match_result.false_positives
    false_negatives = match_result.false_negatives

    matched_classifiable = int(
        sum(
            1
            for aami_class in matched_detection_class.values()
            if aami_class not in excluded_labels
        )
    )
    unsupported_removed = true_positives - matched_classifiable

    # Non-seed detections whose window fell outside the signal.
    boundary_removed = int(np.sum(~within_bounds[1:]))
    # The first detection of the record is dropped for insufficient RR history.
    insufficient_rr_removed = 1

    return RecordConversion(
        record_id=record_name,
        patient_id=get_patient_id(record_name),
        lead_name=lead_name,
        sampling_rate=sampling_rate,
        windows=windows,
        rr_features=rr_features,
        labels_for_sequencing=labels_for_sequencing,
        is_target=is_target,
        is_matched=is_matched,
        detected_samples=detected_sample_per_beat,
        annotation_samples=annotation_sample_per_beat,
        offset_samples=offset_samples,
        offset_ms=offset_ms,
        symbols=symbols,
        aami_labels=aami_labels,
        num_annotations=int(annotation_samples.size),
        num_detections=int(detected_samples.size),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        matched_classifiable=matched_classifiable,
        unsupported_removed=unsupported_removed,
        boundary_removed=boundary_removed,
        insufficient_rr_removed=insufficient_rr_removed,
    )


# ---------------------------------------------------------------------
#                        Assemble Sequence Dataset
# ---------------------------------------------------------------------


def build_xqrs_centered_dataset(
    record_names: list[str],
    detector: RPeakDetector,
    tolerance_ms: float = MATCHING_TOLERANCE_MS,
    sequence_length: int = SEQUENCE_LENGTH,
    normalise_beats: bool = False,
    excluded_labels: set[str] | None = None,
    split_name: str = "validation",
) -> XqrsCenteredDataset:
    """
    Build the complete XQRS-centred sequence dataset for a split.

    The detector is run once per record. Beats are assembled in record
    order, causal sequences are built with the tested sequence builder,
    and only sequences whose final (target) beat is a matched,
    classifiable detection are kept.
    """

    if not record_names:
        raise ValueError("At least one record name must be supplied.")

    if excluded_labels is None:
        excluded_labels = set(EXCLUDED_AAMI_LABELS)

    conversions = [
        build_record_detected_beats(
            record_name=record_name,
            detector=detector,
            tolerance_ms=tolerance_ms,
            normalise_beats=normalise_beats,
            excluded_labels=excluded_labels,
        )
        for record_name in record_names
    ]

    # Concatenate all beat-level arrays in record order and build the
    # contiguous record segments expected by the sequence builder.
    windows_chunks: list[NDArray[np.float64]] = []
    rr_chunks: list[NDArray[np.float64]] = []
    labels_chunks: list[NDArray[np.str_]] = []
    patient_id_values: list[str] = []
    is_target_chunks: list[NDArray[np.bool_]] = []
    is_matched_chunks: list[NDArray[np.bool_]] = []
    detected_chunks: list[NDArray[np.int64]] = []
    annotation_chunks: list[NDArray[np.int64]] = []
    offset_sample_chunks: list[NDArray[np.int64]] = []
    offset_ms_chunks: list[NDArray[np.float64]] = []
    symbol_chunks: list[NDArray[np.str_]] = []
    aami_chunks: list[NDArray[np.str_]] = []
    record_chunks: list[NDArray[np.str_]] = []
    record_segments: list[SequenceSegment] = []

    current_start = 0

    for conversion in conversions:
        num_beats = conversion.num_valid_beats

        if num_beats == 0:
            raise ValueError(
                f"Record {conversion.record_id} produced no valid detected beats."
            )

        windows_chunks.append(conversion.windows)
        rr_chunks.append(conversion.rr_features)
        labels_chunks.append(conversion.labels_for_sequencing)
        patient_id_values.extend([conversion.patient_id] * num_beats)
        is_target_chunks.append(conversion.is_target)
        is_matched_chunks.append(conversion.is_matched)
        detected_chunks.append(conversion.detected_samples)
        annotation_chunks.append(conversion.annotation_samples)
        offset_sample_chunks.append(conversion.offset_samples)
        offset_ms_chunks.append(conversion.offset_ms)
        symbol_chunks.append(conversion.symbols)
        aami_chunks.append(conversion.aami_labels)
        record_chunks.append(np.full(num_beats, conversion.record_id, dtype=object))

        end_index = current_start + num_beats
        record_segments.append(
            {
                "record_id": conversion.record_id,
                "patient_id": conversion.patient_id,
                "start_index": current_start,
                "end_index": end_index,
                "num_sequences": 0,
            }
        )
        current_start = end_index

    windows_all = np.vstack(windows_chunks)
    rr_all = np.vstack(rr_chunks)
    labels_all = np.concatenate(labels_chunks)
    patient_ids_all = np.asarray(patient_id_values, dtype=str)
    is_target_all = np.concatenate(is_target_chunks)
    is_matched_all = np.concatenate(is_matched_chunks)
    detected_all = np.concatenate(detected_chunks)
    annotation_all = np.concatenate(annotation_chunks)
    offset_samples_all = np.concatenate(offset_sample_chunks)
    offset_ms_all = np.concatenate(offset_ms_chunks)
    symbols_all = np.concatenate(symbol_chunks)
    records_all = np.concatenate(record_chunks).astype(str)

    # record_segments needs a num_beats field for validate_dataset.
    segments_for_beats = [
        {
            "record_id": segment["record_id"],
            "patient_id": segment["patient_id"],
            "start_index": segment["start_index"],
            "end_index": segment["end_index"],
            "num_beats": segment["end_index"] - segment["start_index"],
        }
        for segment in record_segments
    ]

    # Reuse the tested causal sequence builder. It only reads the label of
    # the final beat in each window, so context-only beats need no label.
    (
        X_sequences,
        y_sequences,
        rr_sequences,
        sequence_patient_ids,
        target_indices,
        _,
    ) = create_record_sequences(
        X=windows_all,
        y=labels_all,
        rr_features=rr_all,
        patient_ids=patient_ids_all,
        record_metadata=segments_for_beats,
        sequence_length=sequence_length,
    )

    # Keep only sequences whose target beat is a matched, classifiable
    # detection. Every other sequence has a context-only target and is
    # discarded here rather than being given an invented label.
    keep_mask = is_target_all[target_indices]

    final_target_indices = target_indices[keep_mask].astype(np.int64)
    final_X = X_sequences[keep_mask]
    final_rr = rr_sequences[keep_mask]
    final_y = y_sequences[keep_mask].astype(str)
    final_patient_ids = sequence_patient_ids[keep_mask].astype(str)

    # Per-sequence context composition using the four preceding beats.
    context_offsets = np.arange(-(sequence_length - 1), 0)
    context_indices = final_target_indices[:, np.newaxis] + context_offsets
    context_is_matched = is_matched_all[context_indices]
    has_unmatched_context = np.any(~context_is_matched, axis=1)
    num_unmatched_context = np.sum(~context_is_matched, axis=1).astype(np.int64)

    summary = _build_summary(
        record_names=record_names,
        conversions=conversions,
        detector=detector,
        tolerance_ms=tolerance_ms,
        sequence_length=sequence_length,
        normalise_beats=normalise_beats,
        excluded_labels=excluded_labels,
        split_name=split_name,
        num_valid_beats=int(windows_all.shape[0]),
        final_y=final_y,
        has_unmatched_context=has_unmatched_context,
    )

    return XqrsCenteredDataset(
        X_sequences=final_X,
        rr_sequences=final_rr,
        y_labels=final_y,
        patient_ids=final_patient_ids,
        target_indices=final_target_indices,
        audit_records=records_all[final_target_indices],
        audit_detected_samples=detected_all[final_target_indices],
        audit_annotation_samples=annotation_all[final_target_indices],
        audit_offset_samples=offset_samples_all[final_target_indices],
        audit_offset_ms=offset_ms_all[final_target_indices],
        audit_symbols=symbols_all[final_target_indices],
        audit_has_unmatched_context=has_unmatched_context,
        audit_num_unmatched_context=num_unmatched_context,
        summary=summary,
    )


def _build_summary(
    record_names: list[str],
    conversions: list[RecordConversion],
    detector: RPeakDetector,
    tolerance_ms: float,
    sequence_length: int,
    normalise_beats: bool,
    excluded_labels: set[str],
    split_name: str,
    num_valid_beats: int,
    final_y: NDArray[np.str_],
    has_unmatched_context: NDArray[np.bool_],
) -> dict[str, object]:
    """Build the compact JSON-serialisable dataset summary."""

    unique_labels, label_counts = np.unique(final_y, return_counts=True)
    target_class_distribution = {
        str(label): int(count)
        for label, count in zip(unique_labels, label_counts, strict=True)
    }

    return {
        "split_name": split_name,
        "record_names": list(record_names),
        "detector_name": detector.name,
        "detector_config": {"learn": getattr(detector, "learn", None)},
        "tolerance_ms": float(tolerance_ms),
        "sequence_length": int(sequence_length),
        "normalise_beats": bool(normalise_beats),
        "excluded_labels": sorted(excluded_labels),
        "total_annotations": sum(c.num_annotations for c in conversions),
        "total_detections": sum(c.num_detections for c in conversions),
        "true_positives": sum(c.true_positives for c in conversions),
        "false_positives": sum(c.false_positives for c in conversions),
        "false_negatives": sum(c.false_negatives for c in conversions),
        "matched_classifiable_detections": sum(
            c.matched_classifiable for c in conversions
        ),
        "unsupported_labels_removed": sum(c.unsupported_removed for c in conversions),
        "boundary_windows_removed": sum(c.boundary_removed for c in conversions),
        "insufficient_rr_history_removed": sum(
            c.insufficient_rr_removed for c in conversions
        ),
        "num_valid_detected_beats": num_valid_beats,
        "num_final_sequences": int(final_y.shape[0]),
        "num_sequences_with_unmatched_context": int(np.sum(has_unmatched_context)),
        "target_class_distribution": target_class_distribution,
        "per_record": [
            {
                "record_id": c.record_id,
                "lead_name": c.lead_name,
                "annotations": c.num_annotations,
                "detections": c.num_detections,
                "true_positives": c.true_positives,
                "false_positives": c.false_positives,
                "false_negatives": c.false_negatives,
                "matched_classifiable": c.matched_classifiable,
                "valid_detected_beats": c.num_valid_beats,
            }
            for c in conversions
        ],
    }


# ---------------------------------------------------------------------
#                          Validate The Dataset
# ---------------------------------------------------------------------


def validate_xqrs_dataset(dataset: XqrsCenteredDataset) -> None:
    """
    Check the assembled dataset before it is saved.

    Confirms that at least one sequence was produced, that every saved
    array (sequences and audit arrays) shares the same first dimension,
    and that all target labels are supported AAMI classes. Record-boundary
    protection and audit alignment are guaranteed by construction; this is
    a defensive final check.
    """

    num_sequences = int(dataset.X_sequences.shape[0])
    if num_sequences == 0:
        raise ValueError("The XQRS-centred dataset contains no sequences.")

    aligned_arrays = {
        "rr_sequences": dataset.rr_sequences,
        "y_labels": dataset.y_labels,
        "patient_ids": dataset.patient_ids,
        "target_indices": dataset.target_indices,
        "audit_records": dataset.audit_records,
        "audit_detected_samples": dataset.audit_detected_samples,
        "audit_annotation_samples": dataset.audit_annotation_samples,
        "audit_offset_samples": dataset.audit_offset_samples,
        "audit_offset_ms": dataset.audit_offset_ms,
        "audit_symbols": dataset.audit_symbols,
        "audit_has_unmatched_context": dataset.audit_has_unmatched_context,
        "audit_num_unmatched_context": dataset.audit_num_unmatched_context,
    }
    for name, array in aligned_arrays.items():
        if array.shape[0] != num_sequences:
            raise ValueError(
                f"Array '{name}' has first dimension {array.shape[0]}, "
                f"expected {num_sequences}."
            )

    unsupported = set(np.unique(dataset.y_labels).tolist()) - SUPPORTED_AAMI_CLASSES
    if unsupported:
        raise ValueError(
            f"Target labels contain unsupported classes: {sorted(unsupported)}."
        )


# ---------------------------------------------------------------------
#                          Save The Dataset
# ---------------------------------------------------------------------


def save_xqrs_centered_dataset(
    dataset: XqrsCenteredDataset,
    output_dir: Path,
) -> None:
    """
    Save the XQRS-centred sequence arrays, audit arrays, and summary.

    The sequence arrays use the same file names as the expert-centred
    splits so the dataset can be loaded by ECGSequenceDataset. Audit
    arrays are saved as compact ``.npy`` files rather than large JSON.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Core arrays consumed by ECGSequenceDataset and the matched pipeline.
    np.save(output_dir / "X.npy", dataset.X_sequences)
    np.save(output_dir / "rr_features.npy", dataset.rr_sequences)
    np.save(output_dir / "y.npy", dataset.y_labels)
    np.save(output_dir / "patient_ids.npy", dataset.patient_ids)
    np.save(output_dir / "target_indices.npy", dataset.target_indices)

    # Compact audit arrays, aligned row-for-row with the saved sequences.
    np.save(output_dir / "audit_records.npy", dataset.audit_records)
    np.save(output_dir / "audit_detected_samples.npy", dataset.audit_detected_samples)
    np.save(
        output_dir / "audit_annotation_samples.npy",
        dataset.audit_annotation_samples,
    )
    np.save(output_dir / "audit_offset_samples.npy", dataset.audit_offset_samples)
    np.save(output_dir / "audit_offset_ms.npy", dataset.audit_offset_ms)
    np.save(output_dir / "audit_symbols.npy", dataset.audit_symbols)
    np.save(
        output_dir / "audit_has_unmatched_context.npy",
        dataset.audit_has_unmatched_context,
    )
    np.save(
        output_dir / "audit_num_unmatched_context.npy",
        dataset.audit_num_unmatched_context,
    )

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(dataset.summary, file, indent=4)

    logger.info("Saved XQRS-centred dataset to %s", output_dir)


def build_and_save_xqrs_centered_dataset(
    record_names: list[str],
    output_dir: Path,
    detector: RPeakDetector | None = None,
    tolerance_ms: float = MATCHING_TOLERANCE_MS,
    sequence_length: int = SEQUENCE_LENGTH,
    normalise_beats: bool = False,
    split_name: str = "validation",
) -> XqrsCenteredDataset:
    """Build and persist the XQRS-centred dataset for a split."""

    if detector is None:
        detector = XQRSDetector(learn=True)

    dataset = build_xqrs_centered_dataset(
        record_names=record_names,
        detector=detector,
        tolerance_ms=tolerance_ms,
        sequence_length=sequence_length,
        normalise_beats=normalise_beats,
        split_name=split_name,
    )

    validate_xqrs_dataset(dataset)

    save_xqrs_centered_dataset(dataset=dataset, output_dir=output_dir)

    return dataset


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an XQRS-centred sequence dataset (train or val split)."
    )

    parser.add_argument(
        "--split-name",
        choices=list(ALLOWED_SPLIT_NAMES),
        default="val",
        help="Which split to build (train, val, or test).",
    )
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=Path("data/splits_sequences_matched/split_summary_metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("Output directory. Defaults to data/splits_sequences_xqrs/<split-name>."),
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=MATCHING_TOLERANCE_MS,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--normalise-beats",
        action="store_true",
        help="Apply per-beat z-score normalisation (must match the model).",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    record_names = load_split_record_names(args.split_summary_path, args.split_name)

    # Safety: the train, validation, and test record sets must be disjoint.
    assert_splits_pairwise_disjoint(args.split_summary_path)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("data/splits_sequences_xqrs") / args.split_name

    logger.info(
        "Building %r split (%d records) into %s: %s",
        args.split_name,
        len(record_names),
        output_dir,
        record_names,
    )

    dataset = build_and_save_xqrs_centered_dataset(
        record_names=record_names,
        output_dir=output_dir,
        tolerance_ms=args.tolerance_ms,
        sequence_length=args.sequence_length,
        normalise_beats=args.normalise_beats,
        split_name=args.split_name,
    )

    summary = dataset.summary
    logger.info(
        "Split %s summary | records: %d | annotations: %d | detections: %d | "
        "TP: %d | FP: %d | FN: %d | final sequences: %d | "
        "unmatched-context sequences: %d | class distribution: %s",
        args.split_name,
        len(summary["record_names"]),
        summary["total_annotations"],
        summary["total_detections"],
        summary["true_positives"],
        summary["false_positives"],
        summary["false_negatives"],
        summary["num_final_sequences"],
        summary["num_sequences_with_unmatched_context"],
        summary["target_class_distribution"],
    )


if __name__ == "__main__":
    main()
