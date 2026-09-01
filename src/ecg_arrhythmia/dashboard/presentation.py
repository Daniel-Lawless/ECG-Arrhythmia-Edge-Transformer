import math
from collections.abc import Sequence
from dataclasses import dataclass

from ecg_arrhythmia.data.label_mapping import CLASS_LABELS, NUM_CLASSES

UNAVAILABLE = "—"

# Browser-side polling cadence of the single persistent client-side
# live dashboard component (matching the 36-sample / 360 Hz chunk
# cadence), defined here rather than app.py so tests can assert it
# without executing the Streamlit page. Measurement showed Streamlit
# fragment updates being applied too slowly in the browser, so no
# dynamic panel renders through fragments any more - every live value
# updates on this one poll. A presentation cadence only, never a
# real-time guarantee.
ECG_REFRESH_SECONDS = 0.1

# Minimum on-screen time for each queued prediction in the sequential
# presentation panels (classification, model-output bars, recent-beat
# strip). Bursts of predictions are presented one at a time at this
# pace; ECG markers are never delayed by it. Purely a presentation
# hold - it must never be quoted as inference or pipeline latency.
PREDICTION_PRESENTATION_SECONDS = 0.6


# ---------------------------------------------------------------------
#                         Display Mode
# ---------------------------------------------------------------------

# How the Classification and Model output panels choose which
# prediction to show. Both modes read the SAME real-time pipeline: the
# difference is presentation only, which is why neither is called
# "real-time".
#
# Both modes use the identical sequential FIFO presentation; the mode
# changes only the dwell:
#
# Presentation: each prediction is held for
#   PREDICTION_PRESENTATION_SECONDS so a human can read it, with later
#   predictions queued behind it.
# Live: the same queue advanced at LIVE_HOLD_SECONDS, so the display
#   catches up to the stream quickly while still stepping visibly
#   through each prediction.
DISPLAY_MODE_PRESENTATION = "Presentation"
DISPLAY_MODE_LIVE = "Live"

DISPLAY_MODES = (DISPLAY_MODE_PRESENTATION, DISPLAY_MODE_LIVE)

DEFAULT_DISPLAY_MODE = DISPLAY_MODE_PRESENTATION

DISPLAY_MODE_HELP = (
    "Presentation holds predictions for readability. "
    "Live always displays the newest prediction. "
    "Both use the same real-time edge inference; only the dashboard "
    "display differs."
)


# The dwell in live mode. Both modes run the SAME sequential FIFO
# presentation; the mode selects only how long each prediction is held
# before the next one is presented. At 0.1 s the display tracks the
# stream closely while still advancing one prediction at a time, so
# each transition is a separate browser paint rather than several
# predictions collapsing into one.
LIVE_HOLD_SECONDS = 0.1


def hold_seconds_for_mode(mode: str) -> float:
    """
    The per-prediction dwell for a display mode.

    The entire behavioural difference between the modes. An
    unrecognised value falls back to the readable presentation hold
    rather than silently speeding the dashboard up.
    """

    if mode == DISPLAY_MODE_LIVE:
        return LIVE_HOLD_SECONDS

    return PREDICTION_PRESENTATION_SECONDS


# ---------------------------------------------------------------------
#                        Class Display Metadata
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ClassDisplay:
    label: str
    name: str
    colour: str


# Expanded names follow the AAMI grouping comments in
# data/label_mapping.py exactly. Categorical colours, deliberately not
# a green-amber-red severity scale.
CLASS_DISPLAY = {
    "N": ClassDisplay("N", "Normal / bundle branch block", "#4C78A8"),
    "S": ClassDisplay("S", "Supraventricular ectopic", "#F58518"),
    "V": ClassDisplay("V", "Ventricular ectopic", "#B279A2"),
    "F": ClassDisplay("F", "Fusion", "#72B7B2"),
}

UNKNOWN_CLASS_COLOUR = "#9A9A9A"


def class_display(label: str) -> ClassDisplay:
    """Display metadata for a class label; unknown labels stay safe."""

    known = CLASS_DISPLAY.get(label)

    if known is not None:
        return known

    return ClassDisplay(str(label), "Unknown class", UNKNOWN_CLASS_COLOUR)


# ---------------------------------------------------------------------
#                          Model Output
# ---------------------------------------------------------------------


def stable_softmax(logits) -> tuple[float, ...] | None:
    """
    Numerically stable softmax over the wire logits.

    Presentation-only: the stored logits are never modified, and the
    result is a set of softmax-normalised class scores, not calibrated
    probabilities. Returns None for malformed input (wrong length,
    non-numeric, non-finite) rather than crashing the dashboard.
    """

    if logits is None:
        return None

    try:
        values = [float(value) for value in logits]
    except (TypeError, ValueError):
        return None

    if len(values) != NUM_CLASSES:
        return None

    if not all(math.isfinite(value) for value in values):
        return None

    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)

    if total <= 0 or not math.isfinite(total):
        return None

    return tuple(value / total for value in exponentials)


def latest_prediction(recent_predictions: Sequence):
    """The most recently received prediction, or None before any."""

    if not recent_predictions:
        return None

    return recent_predictions[-1]


def recent_beats(recent_predictions: Sequence, limit: int = 12) -> tuple:
    """
    The latest `limit` predictions, oldest to newest, source untouched.
    """

    if limit < 1:
        return ()

    return tuple(recent_predictions[-limit:])


# ---------------------------------------------------------------------
#                             Rhythm
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RhythmEstimate:
    rr_seconds: float | None
    hr_bpm: float | None


_UNAVAILABLE_RHYTHM = RhythmEstimate(rr_seconds=None, hr_bpm=None)


def rhythm_from_prediction(prediction, sampling_rate) -> RhythmEstimate:
    """
    RR and estimated HR behind the latest classified beat.

    peak_indices holds the ordered absolute R-peaks of the sequence's
    beats with the target beat last (see SequenceAssembler), so the
    latest RR is peak_indices[-1] - peak_indices[-2]. Unavailable
    inputs (no prediction, missing rate, fewer than two peaks,
    non-positive RR) yield an unavailable estimate rather than an
    error, and implausible values are never clamped.
    """

    if prediction is None or sampling_rate is None or sampling_rate <= 0:
        return _UNAVAILABLE_RHYTHM

    peaks = prediction.peak_indices

    if len(peaks) < 2:
        return _UNAVAILABLE_RHYTHM

    rr_samples = peaks[-1] - peaks[-2]

    if rr_samples <= 0:
        return _UNAVAILABLE_RHYTHM

    rr_seconds = rr_samples / sampling_rate

    return RhythmEstimate(rr_seconds=rr_seconds, hr_bpm=60.0 / rr_seconds)


# ---------------------------------------------------------------------
#                            Formatting
# ---------------------------------------------------------------------


def format_optional(value, format_spec: str, suffix: str = "") -> str:
    """Format a nullable value; unavailable renders as an em dash."""

    if value is None:
        return UNAVAILABLE

    return f"{value:{format_spec}}{suffix}"


def format_age_seconds(age_seconds: float | None) -> str:
    return format_optional(age_seconds, ".2f", " s")


def format_temperature(temperature_c: float | None) -> str:
    return format_optional(temperature_c, ".1f", " °C")


def format_frequency_mhz(frequency_mhz: float | None) -> str:
    return format_optional(frequency_mhz, ".0f", " MHz")


def format_core_percent(percent: float | None) -> str:
    return format_optional(percent, ".1f", "%")


def format_mib(value_mib: float | None) -> str:
    return format_optional(value_mib, ".0f", " MiB")


def format_milliseconds(value_ms: float | None) -> str:
    return format_optional(value_ms, ".1f", " ms")


def record_time_seconds(sample_index, sampling_rate) -> float | None:
    """Record-relative time for a sample index, None when underivable."""

    if sample_index is None or sampling_rate is None or sampling_rate <= 0:
        return None

    return sample_index / sampling_rate


# ---------------------------------------------------------------------
#                        Status Presentation
# ---------------------------------------------------------------------

_CONNECTION_LABELS = {
    "connected": "Connected",
    "listening": "Listening for Pi",
    "disconnected": "Disconnected",
}

_CONNECTION_COLOURS = {
    "connected": "#2E9E4F",
    "listening": "#D9A21B",
    "disconnected": "#9A9A9A",
}


def connection_label(connection_status: str) -> str:
    return _CONNECTION_LABELS.get(connection_status, connection_status)


def connection_colour(connection_status: str) -> str:
    return _CONNECTION_COLOURS.get(connection_status, UNKNOWN_CLASS_COLOUR)


def condition_text(runtime_condition_active: bool | None) -> str:
    """
    Three-valued power/thermal health text.

    Device/runtime health presentation only - never a clinical alarm.
    """

    if runtime_condition_active is None:
        return "Unknown"

    return "Warning" if runtime_condition_active else "OK"


def yes_no(flag: bool | None) -> str:
    if flag is None:
        return UNAVAILABLE

    return "Yes" if flag else "No"


# Re-exported for the plotting/UI layers so class ordering has one
# source of truth (the project label mapping).
CLASS_ORDER = tuple(CLASS_LABELS)
