from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PeakMatchResult:
    """
    Result of one-to-one matching between expert annotations and
    detected R-peaks.

    The matched and unmatched index arrays contain positions within
    the original annotation and detection arrays.
    """

    matched_annotation_indices: NDArray[np.int64]
    matched_detection_indices: NDArray[np.int64]
    unmatched_annotation_indices: NDArray[np.int64]
    unmatched_detection_indices: NDArray[np.int64]
    offsets_samples: NDArray[np.int64]

    @property
    def true_positives(self) -> int:
        """Number of matched annotation-detection pairs."""
        return int(self.matched_annotation_indices.size)

    @property
    def false_negatives(self) -> int:
        """Number of expert annotations with no matching detection."""
        return int(self.unmatched_annotation_indices.size)

    @property
    def false_positives(self) -> int:
        """Number of detections with no matching expert annotation."""
        return int(self.unmatched_detection_indices.size)


def match_r_peaks(
    annotation_samples: NDArray[np.integer],
    detected_samples: NDArray[np.integer],
    tolerance_samples: int,
) -> PeakMatchResult:
    """
    Perform chronological one-to-one matching between expert heartbeat
    annotations and detected R-peaks.

    An annotation and detection are matched when they differ by no more
    than tolerance_samples. Each annotation can only be mapped to one
    detection
    """

    # expert samples indices
    annotation_array = _validate_sample_indices(
        annotation_samples,
        name="Annotation samples",
    )
    # detected sample indices
    detection_array = _validate_sample_indices(
        detected_samples,
        name="Detected samples",
    )

    # If tolerance samples is not an int or negative, throw an error
    if not isinstance(
        tolerance_samples,
        (int, np.integer),
    ):
        raise TypeError("Tolerance must be an integer number of samples.")

    if tolerance_samples < 0:
        raise ValueError("Tolerance must not be negative.")

    matched_annotation_indices: list[int] = []
    matched_detection_indices: list[int] = []
    unmatched_annotation_indices: list[int] = []
    unmatched_detection_indices: list[int] = []
    offsets_samples: list[int] = []

    # Intialise two pointer
    annotation_index = 0
    detection_index = 0

    # Both arrays are ordered chronologically, so we can move through
    # them once using two pointers.
    while (
        # Stops when either array has been fully processed
        annotation_index < annotation_array.size
        and detection_index < detection_array.size
    ):
        annotation_sample = annotation_array[annotation_index]
        detection_sample = detection_array[detection_index]

        # Positive means the detector overshot, negative means it undershot.
        offset = int(detection_sample - annotation_sample)

        if abs(offset) <= tolerance_samples:
            # The current annotation and detection form one valid pair.
            matched_annotation_indices.append(annotation_index)
            matched_detection_indices.append(detection_index)
            offsets_samples.append(offset)

            # Neither sample can be used again.
            annotation_index += 1
            detection_index += 1

        elif detection_sample < annotation_sample:
            # This detection is already too early for the current
            # annotation. Because later annotations occur even further
            # in the future, it cannot match any of them either.
            unmatched_detection_indices.append(detection_index)
            detection_index += 1

        else:
            # The current annotation is already too early for this
            # detection. Because later detections occur even further in
            # the future, this annotation cannot match any of them.
            unmatched_annotation_indices.append(annotation_index)
            annotation_index += 1

    # Any annotations left after the detections array has been
    # fully processed, they are unmatched, and hence false negatives.
    unmatched_annotation_indices.extend(range(annotation_index, annotation_array.size))

    # Any detections left after the annotation array has been
    # fully processed, are wrong predictions, and hence false positives.
    unmatched_detection_indices.extend(range(detection_index, detection_array.size))

    # Return this information as a dataclass object
    return PeakMatchResult(
        matched_annotation_indices=np.asarray(
            matched_annotation_indices,
            dtype=np.int64,
        ),
        matched_detection_indices=np.asarray(
            matched_detection_indices,
            dtype=np.int64,
        ),
        unmatched_annotation_indices=np.asarray(
            unmatched_annotation_indices,
            dtype=np.int64,
        ),
        unmatched_detection_indices=np.asarray(
            unmatched_detection_indices,
            dtype=np.int64,
        ),
        offsets_samples=np.asarray(
            offsets_samples,
            dtype=np.int64,
        ),
    )


def _validate_sample_indices(
    sample_indices: NDArray[np.integer],
    name: str,
) -> NDArray[np.int64]:
    """Validate and normalise an array of ECG sample positions."""

    sample_array = np.asarray(sample_indices)

    if sample_array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, but received shape {sample_array.shape}."
        )

    # Empty annotation or detection arrays are valid.
    if sample_array.size == 0:
        return np.empty(0, dtype=np.int64)

    if not np.issubdtype(sample_array.dtype, np.integer):
        raise ValueError(f"{name} must contain only integers.")

    sample_array = sample_array.astype(np.int64, copy=False)

    if np.any(sample_array < 0):
        raise ValueError(f"{name} must not contain negative positions.")

    if np.any(np.diff(sample_array) <= 0):
        raise ValueError(
            f"{name} must be strictly increasing and contain no duplicates."
        )

    return sample_array
