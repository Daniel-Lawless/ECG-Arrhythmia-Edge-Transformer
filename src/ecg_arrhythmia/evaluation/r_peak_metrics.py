from dataclasses import dataclass

import numpy as np

from ecg_arrhythmia.evaluation.r_peak_matching import PeakMatchResult


# dataclass is a way to store data. Frozen=True makes it so
# we cannot modify the fields, essentially making it read-only
@dataclass(frozen=True)
class RPeakMetrics:
    """Summary metrics for R-peak detection performance."""

    num_annotations: int
    num_detections: int

    true_positives: int
    false_positives: int
    false_negatives: int

    precision: float
    recall: float
    f1: float

    mean_offset_samples: float | None
    mean_absolute_offset_samples: float | None
    median_absolute_offset_samples: float | None
    standard_deviation_offset_samples: float | None
    maximum_absolute_offset_samples: float | None

    mean_offset_ms: float | None
    mean_absolute_offset_ms: float | None
    median_absolute_offset_ms: float | None
    standard_deviation_offset_ms: float | None
    maximum_absolute_offset_ms: float | None


def compute_r_peak_metrics(
    match_result: PeakMatchResult,
    sampling_rate: float,
) -> RPeakMetrics:
    """
    Calculate detection and timing metrics from matched R-peaks.

    The signed offset is defined as:

        detected sample - expert annotation sample

    A negative offset means the detection occurred before the expert
    annotation. A positive offset means it occurred afterward.
    """

    sampling_rate = _validate_sampling_rate(sampling_rate)

    # Extract fp, fp, and fn from our matched results
    true_positives = match_result.true_positives
    false_positives = match_result.false_positives
    false_negatives = match_result.false_negatives

    # Get the true number of annotations and detections
    num_annotations = true_positives + false_negatives
    num_detections = true_positives + false_positives

    # Out of the predicted peak, how many were in range
    precision = _safe_divide(
        true_positives,
        true_positives + false_positives,
    )

    # How many detections did it get correct
    recall = _safe_divide(
        true_positives,
        true_positives + false_negatives,
    )

    # Take the harmonic mean of the precison and recall.
    # This gives more weight to the lower value
    f1 = _safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    offsets = match_result.offsets_samples

    # Timing statistics cannot be calculated when there are no
    # matched annotation-detection pairs.
    if offsets.size == 0:
        return RPeakMetrics(
            num_annotations=num_annotations,
            num_detections=num_detections,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_offset_samples=None,
            mean_absolute_offset_samples=None,
            median_absolute_offset_samples=None,
            standard_deviation_offset_samples=None,
            maximum_absolute_offset_samples=None,
            mean_offset_ms=None,
            mean_absolute_offset_ms=None,
            median_absolute_offset_ms=None,
            standard_deviation_offset_ms=None,
            maximum_absolute_offset_ms=None,
        )

    offsets_float = offsets.astype(np.float64)
    absolute_offsets = np.abs(offsets_float)

    # One sample represents this many milliseconds.
    milliseconds_per_sample = 1000.0 / sampling_rate

    # Calculate timing metrics
    mean_offset_samples = float(np.mean(offsets_float))
    mean_absolute_offset_samples = float(np.mean(absolute_offsets))
    median_absolute_offset_samples = float(np.median(absolute_offsets))
    standard_deviation_offset_samples = float(np.std(offsets_float))
    maximum_absolute_offset_samples = float(np.max(absolute_offsets))

    return RPeakMetrics(
        num_annotations=num_annotations,
        num_detections=num_detections,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_offset_samples=mean_offset_samples,
        mean_absolute_offset_samples=mean_absolute_offset_samples,
        median_absolute_offset_samples=median_absolute_offset_samples,
        standard_deviation_offset_samples=standard_deviation_offset_samples,
        maximum_absolute_offset_samples=maximum_absolute_offset_samples,
        mean_offset_ms=mean_offset_samples * milliseconds_per_sample,
        mean_absolute_offset_ms=(
            mean_absolute_offset_samples * milliseconds_per_sample
        ),
        median_absolute_offset_ms=(
            median_absolute_offset_samples * milliseconds_per_sample
        ),
        standard_deviation_offset_ms=(
            standard_deviation_offset_samples * milliseconds_per_sample
        ),
        maximum_absolute_offset_ms=(
            maximum_absolute_offset_samples * milliseconds_per_sample
        ),
    )


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Perform division while safely handling a zero denominator."""

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def _validate_sampling_rate(
    sampling_rate: float,
) -> float:
    """Validate and normalise the ECG sampling rate."""
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as error:
        raise TypeError("Sampling rate must be numeric.") from error

    if not np.isfinite(sampling_rate):
        raise ValueError("Sampling rate must be finite.")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    return sampling_rate
