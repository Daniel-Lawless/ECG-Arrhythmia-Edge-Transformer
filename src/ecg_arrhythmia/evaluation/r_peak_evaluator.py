from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.data.load_record import (
    load_record,
    select_signal_channel,
)
from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.evaluation.r_peak_matching import match_r_peaks
from ecg_arrhythmia.evaluation.r_peak_metrics import (
    RPeakMetrics,
    compute_r_peak_metrics,
)
from ecg_arrhythmia.preprocessing.beat_extraction import BEAT_SYMBOLS


@dataclass(frozen=True)
class SymbolDetectionMetrics:
    """Detection results for one expert annotation symbol."""

    annotations: int
    matched: int
    missed: int
    recall: float


@dataclass(frozen=True)
class RecordRPeakEvaluation:
    """R-peak detection results for one complete ECG record."""

    record_name: str
    detector_name: str
    lead_name: str
    sampling_rate: float

    signal_length: int
    signal_duration_seconds: float

    tolerance_samples: int
    tolerance_ms: float

    detector_runtime_seconds: float
    processing_speedup: float

    metrics: RPeakMetrics
    symbol_metrics: dict[str, SymbolDetectionMetrics]


@dataclass(frozen=True)
class DatasetRPeakMetrics:
    """Aggregate detection metrics across multiple ECG records."""

    num_records: int
    num_annotations: int
    num_detections: int

    true_positives: int
    false_positives: int
    false_negatives: int

    precision: float
    recall: float
    f1: float

    mean_offset_ms: float | None
    mean_absolute_offset_ms: float | None
    median_absolute_offset_ms: float | None
    standard_deviation_offset_ms: float | None
    maximum_absolute_offset_ms: float | None


@dataclass(frozen=True)
class DatasetRPeakEvaluation:
    """Complete R-peak evaluation for a collection of records."""

    detector_name: str
    split_name: str
    tolerance_ms: float

    total_signal_duration_seconds: float
    total_detector_runtime_seconds: float
    processing_speedup: float

    metrics: DatasetRPeakMetrics
    symbol_metrics: dict[str, SymbolDetectionMetrics]
    records: list[RecordRPeakEvaluation]


def evaluate_r_peak_record(
    detector: RPeakDetector,
    record_name: str,
    tolerance_ms: float,
) -> RecordRPeakEvaluation:
    """
    Evaluate one R-peak detector on one complete MIT-BIH record.

    The beat-level offsets are used internally while calculating the
    metrics but are not included in the returned evaluation result.
    """

    record_evaluation, _ = _evaluate_r_peak_record(
        detector=detector,
        record_name=record_name,
        tolerance_ms=tolerance_ms,
    )

    return record_evaluation


def evaluate_r_peak_records(
    detector: RPeakDetector,
    record_names: list[str],
    tolerance_ms: float,
    split_name: str,
) -> DatasetRPeakEvaluation:
    """
    Evaluate one R-peak detector across multiple complete ECG records.
    """

    if not record_names:
        raise ValueError("At least one record name must be supplied.")

    # Each internal evaluation returns a list of:
    #
    # 1. The per-record evaluation result.
    # 2. The temporary array of beat-level timing offsets.
    evaluated_records = [
        _evaluate_r_peak_record(
            detector=detector,
            record_name=record_name,
            tolerance_ms=tolerance_ms,
        )
        for record_name in record_names
    ]

    # Retain only the public -level evaluation objects.
    record_results = [record_evaluation for record_evaluation, _ in evaluated_records]

    # This is the total annotations across each record in record_results
    total_annotations = sum(result.metrics.num_annotations for result in record_results)

    # This is the total detections across each record in record_results
    total_detections = sum(result.metrics.num_detections for result in record_results)

    # This is the sum of the tp across the each record in record_results
    total_true_positives = sum(
        result.metrics.true_positives for result in record_results
    )

    # This is the sum of the fp across the each record in record_results
    total_false_posititives = sum(
        result.metrics.false_positives for result in record_results
    )

    # # This is the sum of the fn across the each record in record_results
    total_false_negatives = sum(
        result.metrics.false_negatives for result in record_results
    )

    # Calculate micro-averaged detection metrics from the total counts.
    precision = _safe_divide(
        total_true_positives,
        total_true_positives + total_false_posititives,
    )

    recall = _safe_divide(
        total_true_positives,
        total_true_positives + total_false_negatives,
    )

    f1 = _safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    # Combine all temporary beat-level offsets into one array.
    #
    # This ensures that timing statistics are calculated across every
    # matched beat rather than averaging the per-record averages.
    offset_arrays = [offsets_ms for _, offsets_ms in evaluated_records]

    all_offsets_ms = np.concatenate(offset_arrays)

    timing_metrics = _calculate_aggregate_timing_metrics(
        offsets_ms=all_offsets_ms,
    )

    aggregate_metrics = DatasetRPeakMetrics(
        num_records=len(record_results),
        num_annotations=total_annotations,
        num_detections=total_detections,
        true_positives=total_true_positives,
        false_positives=total_false_posititives,
        false_negatives=total_false_negatives,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        # This unpacks the timing_metrics dict into this object.
        # The keywords in this dict must match what this dataclass
        # expects
        **timing_metrics,
    )

    # Sums the signal duration for each record in record_results to give
    # the total signal duration over these records
    total_signal_duration_seconds = sum(
        result.signal_duration_seconds for result in record_results
    )
    # Sums the detector runtime duration for each record in
    # record_results to give the total detector runtime duration
    # over these records
    total_detector_runtime_seconds = sum(
        result.detector_runtime_seconds for result in record_results
    )

    # Calculates the total processing speedup
    processing_speedup = _safe_divide(
        total_signal_duration_seconds,
        total_detector_runtime_seconds,
    )

    # Calculations the annotation count, matched annotations,
    # missed annotations, and recall for each class across
    # all records in record_results.
    aggregate_symbol_metrics = _aggregate_symbol_metrics(
        record_results=record_results,
    )

    return DatasetRPeakEvaluation(
        detector_name=detector.name,
        split_name=split_name,
        tolerance_ms=round(float(tolerance_ms), 4),
        total_signal_duration_seconds= round(total_signal_duration_seconds, 4),
        total_detector_runtime_seconds=round(total_detector_runtime_seconds,4),
        processing_speedup=round(processing_speedup, 4),
        metrics=aggregate_metrics,
        symbol_metrics=aggregate_symbol_metrics,
        records=record_results,
    )


def _evaluate_r_peak_record(
    detector: RPeakDetector,
    record_name: str,
    tolerance_ms: float,
) -> tuple[RecordRPeakEvaluation, NDArray[np.float64]]:
    """
    Evaluate one complete ECG record.

    Returns the public record-level result and the temporary beat-level
    offsets used when calculating dataset-wide timing statistics.
    """

    tolerance_ms = _validate_tolerance_ms(tolerance_ms)

    # Load the continuous two-channel ECG and expert annotations.
    signals, fields, annotation = load_record(record_name)

    # Use MLII where available, matching the dataset pipeline.
    signal, lead_name = select_signal_channel(
        signals=signals,
        fields=fields,
    )

    sampling_rate = float(fields["fs"])

    if annotation.symbol is None:
        raise ValueError(f"Record {record_name} contains no annotation symbols.")

    # Retain only annotations representing actual heartbeats.
    heartbeat_annotations = [
        (sample, symbol)
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

    # Measure only the R-peak detector runtime.
    #
    # Record loading, matching, and metric calculation are excluded
    # because they are not part of detector inference.
    start_time = perf_counter()

    detected_samples = detector.detect(
        signal=signal,
        sampling_rate=sampling_rate,
    )

    # This is how fast the detector can run through this signal
    detector_runtime_seconds = perf_counter() - start_time

    # Convert the tolerance_ms into samples
    tolerance_samples = round(tolerance_ms * (sampling_rate / 1000.0))

    # Check for fn, fp, and tp of the detected indices to the expert
    # indices
    match_result = match_r_peaks(
        annotation_samples=annotation_samples,
        detected_samples=detected_samples,
        tolerance_samples=tolerance_samples,
    )

    # computes precision, recall, f1, and timing metrics
    # using the matched peaks
    metrics = compute_r_peak_metrics(
        match_result=match_result,
        sampling_rate=sampling_rate,
    )

    # Retrieve the expert symbols belonging to matched and missed beats.
    matched_symbols = annotation_symbols[match_result.matched_annotation_indices]

    # Retrieve the expert symbols belonging to the unmatched beats
    missed_symbols = annotation_symbols[match_result.unmatched_annotation_indices]

    # Creates a dictionary where each key is a class label and its value
    # is a SymbolDetectionMetrics that tells us how often that class appeared
    # in the original annotation set, how many times it matched a detection,
    # how many expert annotations were missed, and its recall.
    symbol_metrics = _build_symbol_metrics(
        annotation_symbols=annotation_symbols,
        matched_symbols=matched_symbols,
        missed_symbols=missed_symbols,
    )

    signal_duration_seconds = len(signal) / sampling_rate

    # The processing speedup is how much faster can the detector run
    # through the signal compared to real time.
    processing_speedup = _safe_divide(
        signal_duration_seconds,
        detector_runtime_seconds,
    )

    # Convert the signed offsets from samples into milliseconds.
    #
    # These are returned separately for temporary aggregation and are
    # not stored in RecordRPeakEvaluation.
    offsets_ms = (
        match_result.offsets_samples.astype(np.float64) * 1000.0 / sampling_rate
    )

    record_evaluation = RecordRPeakEvaluation(
        record_name=record_name,
        detector_name=detector.name,
        lead_name=lead_name,
        sampling_rate=sampling_rate,
        signal_length=len(signal),
        signal_duration_seconds=round(signal_duration_seconds, 4),
        tolerance_samples=tolerance_samples,
        tolerance_ms=round(tolerance_ms, 4),
        detector_runtime_seconds=round(detector_runtime_seconds, 4),
        processing_speedup=round(processing_speedup, 4),
        metrics=metrics,
        symbol_metrics=symbol_metrics,
    )

    return record_evaluation, offsets_ms


def _build_symbol_metrics(
    annotation_symbols: NDArray[np.str_],
    matched_symbols: NDArray[np.str_],
    missed_symbols: NDArray[np.str_],
) -> dict[str, SymbolDetectionMetrics]:
    """Calculate matched and missed counts for each expert symbol."""

    # Counts how often each class occurs in the original expert set
    annotation_counts = Counter(annotation_symbols.tolist())
    # Counts how often each class occurs in the matched set
    matched_counts = Counter(matched_symbols.tolist())
    # Counts how often each class occurs in the unmatched set
    missed_counts = Counter(missed_symbols.tolist())

    # Each symbol gets its own SymbolDetectionMetrics dataclass
    return {
        symbol: SymbolDetectionMetrics(
            annotations=annotation_counts[symbol],
            matched=matched_counts[symbol],
            missed=missed_counts[symbol],
            recall=_safe_divide(
                matched_counts[symbol],
                annotation_counts[symbol],
            ),
        )
        for symbol in sorted(annotation_counts)
    }


def _aggregate_symbol_metrics(
    record_results: list[RecordRPeakEvaluation],
) -> dict[str, SymbolDetectionMetrics]:
    """Aggregate annotation-symbol metrics across all records."""

    annotation_counts: Counter[str] = Counter()
    matched_counts: Counter[str] = Counter()
    missed_counts: Counter[str] = Counter()

    # For each record result
    for result in record_results:
        # For each symbol and SymbolDetectionMetrics
        for symbol, metrics in result.symbol_metrics.items():
            # This gives us the total annotations, matched_counts,
            # and missed counts for each class over these records.
            annotation_counts[symbol] += metrics.annotations
            matched_counts[symbol] += metrics.matched
            missed_counts[symbol] += metrics.missed

    # This will now return a dictionary where the key
    # is the class and the values if a SymbolDetectionMetrics
    # object that will now contain the annotations, match, missed,
    # and recall across all records in record_results.
    return {
        symbol: SymbolDetectionMetrics(
            annotations=annotation_counts[symbol],
            matched=matched_counts[symbol],
            missed=missed_counts[symbol],
            recall=round(_safe_divide(
                matched_counts[symbol],
                annotation_counts[symbol],
            ), 4),
        )
        for symbol in sorted(annotation_counts)
    }


def _calculate_aggregate_timing_metrics(
    offsets_ms: NDArray[np.float64],
) -> dict[str, float | None]:
    """Calculate timing statistics across every matched beat."""

    if offsets_ms.size == 0:
        return {
            "mean_offset_ms": None,
            "mean_absolute_offset_ms": None,
            "median_absolute_offset_ms": None,
            "standard_deviation_offset_ms": None,
            "maximum_absolute_offset_ms": None,
        }

    absolute_offsets_ms = np.abs(offsets_ms)

    return {
        # Signed mean indicates whether the detector tends to be early
        # or late relative to expert annotations.
        "mean_offset_ms": round(float(np.mean(offsets_ms)), 4),
        # Mean absolute offset represents the average timing error
        # regardless of whether the detector was early or late.
        "mean_absolute_offset_ms": round(float(np.mean(absolute_offsets_ms)), 4),
        "median_absolute_offset_ms": round(float(np.median(absolute_offsets_ms)), 4),
        "standard_deviation_offset_ms": round(float(np.std(offsets_ms)), 4),
        "maximum_absolute_offset_ms": round(float(np.max(absolute_offsets_ms)), 4),
    }


def _validate_tolerance_ms(
    tolerance_ms: float,
) -> float:
    """Validate and normalise the matching tolerance."""

    if isinstance(tolerance_ms, bool):
        raise TypeError("Matching tolerance must be numeric.")

    try:
        tolerance_ms = float(tolerance_ms)
    except (TypeError, ValueError) as error:
        raise TypeError("Matching tolerance must be numeric.") from error

    if not np.isfinite(tolerance_ms):
        raise ValueError("Matching tolerance must be finite.")

    if tolerance_ms < 0:
        raise ValueError("Matching tolerance must not be negative.")

    return tolerance_ms


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Return zero when a metric denominator is zero."""

    if denominator == 0:
        return 0.0

    return round(float(numerator / denominator), 4)
