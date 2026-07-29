from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SampleChunk:
    samples: NDArray[np.float64]
    start_index: int
    sampling_rate: float

    # the dataclass does the __init__() boilerplate code for us,
    # __post_init__() is often used to do validation of the constructor
    # arguments. Since frozen=True makes the attributes (fields) immutable
    # object.__setattr__() overrides this so we can define validation functions
    # to apply to our arguments after they have been set
    def __post_init__(self) -> None:
        # The dataclass is frozen, so normal attribute assignment is not
        # allowed. object.__setattr__ lets us store the validated values
        # during construction while keeping the finished object immutable.
        object.__setattr__(
            self,
            # Name of attribute
            "samples",
            # What you want done to this attribute.
            # The value this function returns will be
            # assigned to this the provided attribute
            validate_samples(self.samples),
        )

        object.__setattr__(
            self,
            "start_index",
            validate_start_index(self.start_index),
        )

        object.__setattr__(
            self,
            "sampling_rate",
            validate_sampling_rate(self.sampling_rate),
        )

    @property
    def num_samples(self) -> int:
        """Number of samples contained in this chunk."""

        # Return the number of samples in this chunk.
        return int(self.samples.size)

    @property
    def stop_index(self) -> int:
        """Absolute index one past the final sample in this chunk."""

        # Return the index of the first sample after this chunk.
        return self.start_index + self.num_samples

    @property
    def last_index(self) -> int:
        """Absolute index of the final sample in this chunk."""

        # Return the index of the final sample included in this chunk.
        return self.stop_index - 1

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of ECG represented by this chunk."""

        # Return the duration of ECG represented by this chunk.
        return self.num_samples / self.sampling_rate


def validate_samples(samples: np.ndarray) -> NDArray[np.float64]:

    # Convert the samples to a float64 NumPy array.
    sample_array = np.asarray(samples, dtype=np.float64)

    # Should be a 1 dimensional array of amplitude values
    if sample_array.ndim != 1:
        raise ValueError(
            f"Expected a 1-dimensional array, recieved {sample_array.shape}"
        )

    # All values should be finite
    if not np.all(np.isfinite(sample_array)):
        raise ValueError("All amplitude values should be finite")

    if sample_array.size == 0:
        raise ValueError("Chunk samples must not be empty.")

    # frozen=True ensures we cannot set samples to another array,
    # and setflags(write=False) ensures we cannot change the
    # values WITHIN this array.
    sample_array.setflags(write=False)

    return sample_array


def validate_start_index(start_index: int) -> int:
    """
    Validate and normalise an absolute sample position.

    Shared by the streaming modules so a chunk and the engine agree on
    what a usable absolute sample index is.
    """

    # bool is a subclass of int in Python. Without this explicit check,
    # validate_start_index(False) and validate_start_index(True) would both
    # pass integer validation and become sample indices 0 and 1.
    if isinstance(start_index, bool) or not isinstance(
        start_index,
        int | np.integer,
    ):
        raise TypeError("Chunk start index must be an integer.")

    if start_index < 0:
        raise ValueError("Chunk start index must not be negative.")

    return int(start_index)


def validate_sampling_rate(sampling_rate: float) -> float:

    if isinstance(sampling_rate, bool):
        raise TypeError("Sampling rate must be numeric.")

    # Try to convert the supplied value to a standard Python float.
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as error:
        raise TypeError("Sampling rate must be numeric.") from error

    # Reject NaN and positive or negative infinity.
    if not np.isfinite(sampling_rate):
        raise ValueError("Sampling rate must be finite.")

    # A valid sampling frequency must be greater than zero.
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    return sampling_rate
