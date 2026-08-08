from collections.abc import Sequence

import numpy as np

# Recognised heartbeat annotations. This will exclude non heart beat annotations
# fmt: off
BEAT_SYMBOLS = {
    "N", "L", "R", "e",
    "j", "A", "a", "J", 
    "S", "V", "E", "F",
    "Q", "?", "/", "f",
}
# fmt: on

# Start and end of the window.
SAMPLES_BEFORE = 90
SAMPLES_AFTER = 150

# This will give us a 240 sample window.
WINDOW_SIZE = SAMPLES_BEFORE + SAMPLES_AFTER

# How many amplitudes values are counted a second
SAMPLING_RATE = 360

# RR config
LOCAL_RR_WINDOW = 10
MIN_RR_SECONDS = 0.25
MAX_RR_SECONDS = 3.0
MIN_RR_RATIO = 0.25
MAX_RR_RATIO = 4.0

# Number of beats the trained transformer consumes per input.
SEQUENCE_LENGTH = 5

# previous_rr_seconds and rr_ratio, the two features produced below.
RR_FEATURE_DIM = 2


def previous_rr_seconds(peak_sample: int, previous_peak_sample: int) -> float:

    # This calculates the number of seconds between this peak and the
    # previous peak
    rr_seconds = (peak_sample - previous_peak_sample) / SAMPLING_RATE

    # We then clip the result so unusually small or large intervals
    # cannot be outside of MIN_RR_SECONDS and MAX_RR_SECONDS
    return float(np.clip(rr_seconds, MIN_RR_SECONDS, MAX_RR_SECONDS))


def rr_ratio(prev_rr_seconds: float, recent_rr_seconds: Sequence[float]) -> float:

    # LOCAL_RR_WINDOW is how much intervals we are considering
    # in recent_rr_seconds each element in recent_rr_seconds
    # is a time between 2 peaks.
    local_history = list(recent_rr_seconds)[-LOCAL_RR_WINDOW:]
    # We calculate the mean of the local window. If we haven't seen any
    # time between peaks yet, then we just take prev_rr_seconds to
    # be out mean
    local_mean_rr = np.mean(local_history) if local_history else prev_rr_seconds

    # This returns how this peaks timeframe to the previous peak differs
    # from the mean.
    return float(
        np.clip(
            prev_rr_seconds / (local_mean_rr + 1e-8),
            MIN_RR_RATIO,
            MAX_RR_RATIO,
        )
    )


# Extract beats and labels for a given signal.
def extract_beats(
    signal: np.ndarray,
    annotation_samples: np.ndarray,
    annotation_symbols: list[str],
    normalise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    extracted_beats = []
    extracted_labels = []
    rr_features = []

    previous_heartbeat_sample: int | None = None
    recent_rr_seconds: list[float] = []

    # Creates a 240 sample window around each annotation index.
    # This will form each datapoint. The corresponding label for that
    # window will be its y value. stict=True means they must have the same length
    for sample_index, symbol in zip(
        annotation_samples, annotation_symbols, strict=True
    ):
        # Ignore annotations that do not represent heartbeats
        if symbol not in BEAT_SYMBOLS:
            continue

        # We cannot calculate a previous RR interval for the first beat
        if previous_heartbeat_sample is None:
            previous_heartbeat_sample = sample_index
            continue

        # Define samples before and after beat
        start = sample_index - SAMPLES_BEFORE
        end = sample_index + SAMPLES_AFTER

        # Time from the previous heartbeat, clipped to combat outliers.
        # Every heartbeat advances the RR chain, including the ones whose
        # window is found to be incomplete just below.
        prev_rr = previous_rr_seconds(sample_index, previous_heartbeat_sample)

        previous_heartbeat_sample = sample_index

        # Skip beats that cannot produce a complete 240-sample window
        if start < 0 or end > len(signal):
            continue

        # 240 sample beat
        beat = signal[start:end]

        # Normalise the beat
        if normalise:
            beat = (beat - beat.mean()) / (beat.std() + 1e-8)

        # Compare the current RR interval against the recent rhythm for this record.
        # local_mean_rr is the average RR interval from recent previous beats.
        # Example: if local_mean_rr = 0.80s and prev_rr_seconds = 0.50s,
        # then rr_ratio = 0.50 / 0.80 = 0.625.
        # rr_ratio < 1 means the beat arrived earlier than expected.
        # rr_ratio ~= 1 means the beat arrived around the expected time.
        # rr_ratio > 1 means the beat arrived later than expected.
        # This is useful for S beats because they can look similar to normal beats,
        # but are often premature, so timing can help distinguish them from N.
        ratio = rr_ratio(prev_rr, recent_rr_seconds)

        # Append window and corresponding symbol.
        extracted_beats.append(beat)
        extracted_labels.append(symbol)
        rr_features.append([prev_rr, ratio])

        # Only completed beats contribute to the local rhythm.
        recent_rr_seconds.append(prev_rr)

    # Converts the list of labels into a numpy array
    labels = np.array(extracted_labels)

    if len(extracted_beats) == 0:
        return (
            # Empty numpy array with 0 rows and 240 columns. It will take values
            # that has the same data type as signal.
            np.empty((0, WINDOW_SIZE), dtype=signal.dtype),
            # Empty numpy array for labels
            np.array([], dtype=str),
            # Empty numpy array for rr_features
            np.empty((0, 2), dtype=np.float32),
        )

    # Stacks the list of numpy arrays into a 2d matrix.
    beats_matrix = np.vstack(extracted_beats)
    rr_features = np.array(rr_features)

    return beats_matrix, labels, rr_features
