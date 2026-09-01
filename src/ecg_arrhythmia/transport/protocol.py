import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
    from ecg_arrhythmia.streaming.sample_chunk import SampleChunk

# The version stamped on outgoing messages.
SCHEMA_VERSION = 3

# Versions this receiver still accepts: a v1 sender keeps working
# (without runtime telemetry) and a v2 sender keeps working (without
# the model-stage measurements added by v3).
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)

MESSAGE_TYPE_SAMPLE_CHUNK = "sample_chunk"
MESSAGE_TYPE_PREDICTION = "prediction"
MESSAGE_TYPE_RUNTIME_STATUS = "runtime_status"

# Which message types each schema version defines. sample_chunk and
# prediction mean exactly the same thing in every version; v2 adds
# runtime_status and v3 extends runtime_status with nullable
# model-stage fields (see RUNTIME_STATUS_V3_FIELDS) - v2's strict
# required-field contract is preserved rather than mutated.
MESSAGE_TYPES_BY_VERSION = {
    1: (MESSAGE_TYPE_SAMPLE_CHUNK, MESSAGE_TYPE_PREDICTION),
    2: (
        MESSAGE_TYPE_SAMPLE_CHUNK,
        MESSAGE_TYPE_PREDICTION,
        MESSAGE_TYPE_RUNTIME_STATUS,
    ),
    3: (
        MESSAGE_TYPE_SAMPLE_CHUNK,
        MESSAGE_TYPE_PREDICTION,
        MESSAGE_TYPE_RUNTIME_STATUS,
    ),
}

# Live model-stage measurement fields added to runtime_status by
# schema version 3. All nullable (null = not measured yet, never a
# fabricated zero); when present, latency/throughput must be positive
# finite numbers and the measurement age non-negative. The timing
# seam matches Section 5.2's model-stage boundary (the classifier's
# predict() call); these are live operational readings, not a
# replacement for the controlled Section 5.2 benchmark.
RUNTIME_STATUS_V3_FIELDS = (
    "model_inference_mean_ms",
    "model_throughput_sequences_per_second",
    "model_measurement_age_seconds",
)

ENCODING = "utf-8"
FRAME_DELIMITER = b"\n"

REQUIRED_FIELDS = {
    MESSAGE_TYPE_SAMPLE_CHUNK: (
        "record_name",
        "start_index",
        "sampling_rate",
        "samples",
    ),
    MESSAGE_TYPE_PREDICTION: (
        "record_name",
        "target_peak_index",
        "peak_indices",
        "logits",
        "predicted_class_index",
        "predicted_label",
    ),
    MESSAGE_TYPE_RUNTIME_STATUS: (
        "record_name",
        "latest_sample_index",
        "temperature_c",
        "process_cpu_percent",
        "process_rss_mib",
        "available_ram_mib",
        "cpu_frequency_mhz",
        "cpu_governor",
        "under_voltage_active",
        "frequency_capped_active",
        "throttling_active",
        "soft_temp_limit_active",
        "runtime_condition_occurred",
        "window_max_chunk_processing_ms",
        "window_min_processing_headroom_ms",
    ),
}

# Fields that must decode to JSON arrays for the message to be usable.
LIST_FIELDS = {
    MESSAGE_TYPE_SAMPLE_CHUNK: ("samples",),
    MESSAGE_TYPE_PREDICTION: ("peak_indices", "logits"),
    MESSAGE_TYPE_RUNTIME_STATUS: (),
}

# Fields that must be real (non-null) numbers: the processing metrics
# are always measurable while streaming runs, unlike hardware sources.
NUMERIC_FIELDS = {
    MESSAGE_TYPE_RUNTIME_STATUS: (
        "window_max_chunk_processing_ms",
        "window_min_processing_headroom_ms",
    ),
}


class ProtocolError(ValueError):
    """A wire message is malformed, unsupported or incomplete."""


class TransportError(RuntimeError):
    """
    A transport-level (socket) operation failed.

    Defined here so the sender and receiver share one exception type
    without importing each other; this module itself never touches
    sockets.
    """


def _encode(payload: dict) -> bytes:
    """Compact JSON, UTF-8, exactly one trailing newline."""

    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

    return text.encode(ENCODING) + FRAME_DELIMITER


def encode_sample_chunk(
    chunk: "SampleChunk",
    record_name: str | None,
) -> bytes:
    """One ECG chunk as a wire frame, converting NumPy values here."""

    return _encode(
        {
            "schema_version": SCHEMA_VERSION,
            "message_type": MESSAGE_TYPE_SAMPLE_CHUNK,
            "record_name": record_name,
            "start_index": int(chunk.start_index),
            "sampling_rate": float(chunk.sampling_rate),
            "samples": [float(sample) for sample in chunk.samples],
        }
    )


def encode_prediction(
    event: "PredictionEvent",
    record_name: str | None,
) -> bytes:
    """One prediction event as a wire frame, converting NumPy values here."""

    return _encode(
        {
            "schema_version": SCHEMA_VERSION,
            "message_type": MESSAGE_TYPE_PREDICTION,
            "record_name": record_name,
            "target_peak_index": int(event.target_peak_index),
            "peak_indices": [int(peak) for peak in event.peak_indices],
            "logits": [float(logit) for logit in event.logits],
            "predicted_class_index": int(event.predicted_class_index),
            "predicted_label": str(event.predicted_label),
        }
    )


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value) -> bool | None:
    return None if value is None else bool(value)


def encode_runtime_status(
    record_name: str | None,
    latest_sample_index: int | None,
    temperature_c: float | None,
    process_cpu_percent: float | None,
    process_rss_mib: float | None,
    available_ram_mib: float | None,
    cpu_frequency_mhz: float | None,
    cpu_governor: str | None,
    under_voltage_active: bool | None,
    frequency_capped_active: bool | None,
    throttling_active: bool | None,
    soft_temp_limit_active: bool | None,
    runtime_condition_occurred: bool | None,
    window_max_chunk_processing_ms: float,
    window_min_processing_headroom_ms: float,
    model_inference_mean_ms: float | None,
    model_throughput_sequences_per_second: float | None,
    model_measurement_age_seconds: float | None,
) -> bytes:
    """
    One live edge-telemetry frame (schema version 3).

    Hardware fields are null when their source is unavailable, never
    fabricated zeros. The four current power/thermal conditions are
    transmitted individually - throttling_active is the literal
    throttling bit only - and runtime_condition_occurred aggregates
    the sticky since-boot bits, so a live indicator can neither label
    under-voltage as throttling nor show a historical event as
    current. The two processing metrics are always required:
    window_max_chunk_processing_ms is the largest process_chunk()
    duration in the latest telemetry interval, and the headroom is the
    nominal chunk period minus it (NOT a hard-real-time deadline
    margin; scheduling and network timing are excluded). The three
    model-stage fields (v3) are null before the first-ever timed
    inference, never zero; retained values are dated by
    model_measurement_age_seconds rather than re-labelled as fresh.
    """

    return _encode(
        {
            "schema_version": SCHEMA_VERSION,
            "message_type": MESSAGE_TYPE_RUNTIME_STATUS,
            "record_name": record_name,
            "latest_sample_index": (
                None if latest_sample_index is None else int(latest_sample_index)
            ),
            "temperature_c": _optional_float(temperature_c),
            "process_cpu_percent": _optional_float(process_cpu_percent),
            "process_rss_mib": _optional_float(process_rss_mib),
            "available_ram_mib": _optional_float(available_ram_mib),
            "cpu_frequency_mhz": _optional_float(cpu_frequency_mhz),
            "cpu_governor": None if cpu_governor is None else str(cpu_governor),
            "under_voltage_active": _optional_bool(under_voltage_active),
            "frequency_capped_active": _optional_bool(frequency_capped_active),
            "throttling_active": _optional_bool(throttling_active),
            "soft_temp_limit_active": _optional_bool(soft_temp_limit_active),
            "runtime_condition_occurred": _optional_bool(runtime_condition_occurred),
            "window_max_chunk_processing_ms": float(window_max_chunk_processing_ms),
            "window_min_processing_headroom_ms": float(
                window_min_processing_headroom_ms
            ),
            "model_inference_mean_ms": _optional_float(model_inference_mean_ms),
            "model_throughput_sequences_per_second": _optional_float(
                model_throughput_sequences_per_second
            ),
            "model_measurement_age_seconds": _optional_float(
                model_measurement_age_seconds
            ),
        }
    )


def decode_message(frame: bytes | str) -> dict:
    """
    Decode and validate one frame (without its newline delimiter).

    Returns the message as a plain dict. Raises ProtocolError for
    malformed JSON, non-object payloads, unsupported schema versions,
    unknown message types, message types not defined by the frame's
    schema version, and missing, non-array or non-numeric required
    fields.
    """

    if isinstance(frame, bytes):
        try:
            frame = frame.decode(ENCODING)
        except UnicodeDecodeError as error:
            raise ProtocolError(f"frame is not valid UTF-8: {error}") from error

    try:
        message = json.loads(frame)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"frame is not valid JSON: {error}") from error

    if not isinstance(message, dict):
        raise ProtocolError(
            f"frame must decode to a JSON object, found {type(message).__name__}"
        )

    version = message.get("schema_version")

    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
        raise ProtocolError(
            f"unsupported schema version {version!r}; this receiver "
            f"supports versions {supported}"
        )

    message_type = message.get("message_type")

    if message_type not in REQUIRED_FIELDS:
        raise ProtocolError(f"unknown message type {message_type!r}")

    if message_type not in MESSAGE_TYPES_BY_VERSION[version]:
        raise ProtocolError(
            f"message type {message_type!r} is not defined in schema version {version}"
        )

    required = REQUIRED_FIELDS[message_type]

    if message_type == MESSAGE_TYPE_RUNTIME_STATUS and version >= 3:
        required = required + RUNTIME_STATUS_V3_FIELDS

    missing = [field for field in required if field not in message]

    if missing:
        raise ProtocolError(
            f"{message_type} message is missing required fields: " + ", ".join(missing)
        )

    for field in LIST_FIELDS[message_type]:
        if not isinstance(message[field], list):
            raise ProtocolError(
                f"{message_type} field {field!r} must be a JSON array, "
                f"found {type(message[field]).__name__}"
            )

    for field in NUMERIC_FIELDS.get(message_type, ()):
        value = message[field]

        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProtocolError(
                f"{message_type} field {field!r} must be a number, "
                f"found {type(value).__name__}"
            )

    # v3 model-stage fields: null is a first-class value (never
    # measured), but a present measurement must be a finite number
    # with the right sign - a zero or negative latency/throughput can
    # only be an instrumentation bug, never a real reading.
    if message_type == MESSAGE_TYPE_RUNTIME_STATUS and version >= 3:
        for field in RUNTIME_STATUS_V3_FIELDS:
            value = message[field]

            if value is None:
                continue

            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ProtocolError(
                    f"{message_type} field {field!r} must be null or a "
                    f"number, found {type(value).__name__}"
                )

            if not math.isfinite(value):
                raise ProtocolError(f"{message_type} field {field!r} must be finite")

            if field == "model_measurement_age_seconds":
                if value < 0:
                    raise ProtocolError(
                        f"{message_type} field {field!r} must be null or non-negative"
                    )
            elif value <= 0:
                raise ProtocolError(
                    f"{message_type} field {field!r} must be null or positive"
                )

    return message
