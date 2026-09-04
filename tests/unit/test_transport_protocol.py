import json

import numpy as np
import pytest

from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.transport.protocol import (
    SCHEMA_VERSION,
    ProtocolError,
    decode_message,
    encode_prediction,
    encode_runtime_status,
    encode_sample_chunk,
)

# Amplitudes chosen to be exactly representable in binary floating
# point, so round-trip assertions can use plain equality.
CHUNK_SAMPLES = [0.5, -0.25, 0.125, 2.0]


def _chunk() -> SampleChunk:
    return SampleChunk(
        # NumPy scalar start index and float64 samples: the protocol
        # must convert both at the boundary.
        samples=np.array(CHUNK_SAMPLES, dtype=np.float64),
        start_index=np.int64(72),
        sampling_rate=360.0,
    )


def _event() -> PredictionEvent:
    return PredictionEvent(
        target_peak_index=2358,
        peak_indices=(2000, 2358, 2799),
        logits=np.array([3.5, -1.25, 0.5, -2.0], dtype=np.float32),
        predicted_class_index=0,
        predicted_label="N",
    )


def _mutated_frame(frame: bytes, **overrides) -> str:
    message = json.loads(frame.decode("utf-8"))
    message.update(overrides)

    return json.dumps(message)


# ---------------------------------------------------------------------
#                             Framing
# ---------------------------------------------------------------------


def test_frames_are_compact_utf8_with_exactly_one_trailing_newline():
    for frame in (
        encode_sample_chunk(_chunk(), "114"),
        encode_prediction(_event(), "114"),
    ):
        assert isinstance(frame, bytes)
        assert frame.endswith(b"\n")
        assert frame.count(b"\n") == 1
        # Compact separators: no spaces anywhere in these payloads.
        assert b" " not in frame
        frame.decode("utf-8")


# ---------------------------------------------------------------------
#                       Sample-Chunk Round Trip
# ---------------------------------------------------------------------


def test_sample_chunk_round_trips_with_order_and_values_preserved():
    message = decode_message(encode_sample_chunk(_chunk(), "114"))

    assert message["schema_version"] == SCHEMA_VERSION
    assert message["message_type"] == "sample_chunk"
    assert message["record_name"] == "114"
    assert message["start_index"] == 72
    assert message["sampling_rate"] == 360.0
    assert message["samples"] == CHUNK_SAMPLES


def test_sample_chunk_values_decode_to_plain_python_types():
    message = decode_message(encode_sample_chunk(_chunk(), "114"))

    assert type(message["start_index"]) is int
    assert type(message["sampling_rate"]) is float
    assert all(type(sample) is float for sample in message["samples"])


def test_record_name_may_be_null_for_an_unnamed_stream():
    message = decode_message(encode_sample_chunk(_chunk(), None))

    assert message["record_name"] is None


def test_stop_index_is_not_transmitted_because_it_is_derivable():
    message = decode_message(encode_sample_chunk(_chunk(), "114"))

    assert "stop_index" not in message
    # The receiver reconstructs it from what is transmitted.
    assert message["start_index"] + len(message["samples"]) == 76


# ---------------------------------------------------------------------
#                       Prediction Round Trip
# ---------------------------------------------------------------------


def test_prediction_round_trips_with_all_event_fields_preserved():
    message = decode_message(encode_prediction(_event(), "114"))

    assert message["schema_version"] == SCHEMA_VERSION
    assert message["message_type"] == "prediction"
    assert message["record_name"] == "114"
    assert message["target_peak_index"] == 2358
    assert message["peak_indices"] == [2000, 2358, 2799]
    assert message["predicted_class_index"] == 0
    assert message["predicted_label"] == "N"


def test_float32_logits_convert_exactly_to_json_floats():
    message = decode_message(encode_prediction(_event(), "114"))

    assert message["logits"] == [3.5, -1.25, 0.5, -2.0]
    assert all(type(logit) is float for logit in message["logits"])


# ---------------------------------------------------------------------
#                        Rejected Messages
# ---------------------------------------------------------------------


def test_an_unsupported_schema_version_fails_clearly():
    frame = _mutated_frame(encode_sample_chunk(_chunk(), "114"), schema_version=999)

    with pytest.raises(ProtocolError, match="unsupported schema version 999"):
        decode_message(frame)


def test_an_unknown_message_type_fails_clearly():
    frame = _mutated_frame(
        encode_sample_chunk(_chunk(), "114"),
        message_type="heartbeat",
    )

    with pytest.raises(ProtocolError, match="unknown message type 'heartbeat'"):
        decode_message(frame)


def test_malformed_json_fails_clearly():
    with pytest.raises(ProtocolError, match="not valid JSON"):
        decode_message(b'{"schema_version": 1, ')


def test_a_non_object_payload_fails_clearly():
    with pytest.raises(ProtocolError, match="JSON object"):
        decode_message(b"[1, 2, 3]")


def test_invalid_utf8_fails_clearly():
    with pytest.raises(ProtocolError, match="not valid UTF-8"):
        decode_message(b"\xff\xfe{}")


@pytest.mark.parametrize(
    "field",
    ["record_name", "start_index", "sampling_rate", "samples"],
)
def test_missing_sample_chunk_fields_fail_clearly(field):
    message = json.loads(encode_sample_chunk(_chunk(), "114").decode("utf-8"))
    del message[field]

    with pytest.raises(ProtocolError, match=f"missing required fields: {field}"):
        decode_message(json.dumps(message))


@pytest.mark.parametrize(
    "field",
    ["target_peak_index", "peak_indices", "logits", "predicted_label"],
)
def test_missing_prediction_fields_fail_clearly(field):
    message = json.loads(encode_prediction(_event(), "114").decode("utf-8"))
    del message[field]

    with pytest.raises(ProtocolError, match=f"missing required fields: {field}"):
        decode_message(json.dumps(message))


def test_non_array_samples_fail_clearly():
    frame = _mutated_frame(
        encode_sample_chunk(_chunk(), "114"),
        samples="not-an-array",
    )

    with pytest.raises(ProtocolError, match="'samples' must be a JSON array"):
        decode_message(frame)


# ---------------------------------------------------------------------
#                Runtime Status (Schema Versions 2 and 3)
# ---------------------------------------------------------------------


def _runtime_status_fields(**overrides) -> dict:
    fields = {
        "record_name": "114",
        "latest_sample_index": 46217,
        "temperature_c": 48.7,
        "process_cpu_percent": 3.5,
        "process_rss_mib": 253.0,
        "available_ram_mib": 610.0,
        "cpu_frequency_mhz": 2400.0,
        "cpu_governor": "performance",
        "under_voltage_active": False,
        "frequency_capped_active": False,
        "throttling_active": False,
        "soft_temp_limit_active": False,
        "runtime_condition_occurred": False,
        "window_max_chunk_processing_ms": 1.4,
        "window_min_processing_headroom_ms": 98.6,
        # Live model-stage values chosen NOT to match the Section 5.2
        # benchmark numbers, so no test accidentally pins those.
        "model_inference_mean_ms": 1.41,
        "model_throughput_sequences_per_second": 709.2,
        "model_measurement_age_seconds": 0.0,
    }
    fields.update(overrides)

    return fields


def test_runtime_status_round_trips_with_all_fields_preserved():
    fields = _runtime_status_fields()

    message = decode_message(encode_runtime_status(**fields))

    assert message["schema_version"] == SCHEMA_VERSION == 3
    assert message["message_type"] == "runtime_status"
    assert "hardware_sample_age_seconds" not in message
    assert "hardware_sample_stale" not in message

    for key, value in fields.items():
        assert message[key] == value


@pytest.mark.parametrize("age,stale", [(None, True), (0.0, False), (3.1, True)])
def test_optional_hardware_cache_metadata_round_trips(age, stale):
    message = decode_message(
        encode_runtime_status(
            **_runtime_status_fields(
                hardware_sample_age_seconds=age,
                hardware_sample_stale=stale,
            )
        )
    )

    assert message["hardware_sample_age_seconds"] == age
    assert message["hardware_sample_stale"] is stale


@pytest.mark.parametrize("age", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_hardware_cache_age_is_rejected(age):
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        hardware_sample_age_seconds=age,
    )

    with pytest.raises(ProtocolError, match="hardware_sample_age_seconds"):
        decode_message(frame)


@pytest.mark.parametrize("stale", [0, 1, "true", []])
def test_invalid_hardware_cache_stale_flag_is_rejected(stale):
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        hardware_sample_stale=stale,
    )

    with pytest.raises(ProtocolError, match="hardware_sample_stale"):
        decode_message(frame)


def test_runtime_status_hardware_fields_may_be_null():
    fields = _runtime_status_fields(
        temperature_c=None,
        process_cpu_percent=None,
        process_rss_mib=None,
        available_ram_mib=None,
        cpu_frequency_mhz=None,
        cpu_governor=None,
        under_voltage_active=None,
        frequency_capped_active=None,
        throttling_active=None,
        soft_temp_limit_active=None,
        runtime_condition_occurred=None,
    )

    message = decode_message(encode_runtime_status(**fields))

    assert message["temperature_c"] is None
    assert message["cpu_frequency_mhz"] is None
    assert message["cpu_governor"] is None
    assert message["under_voltage_active"] is None
    assert message["frequency_capped_active"] is None
    assert message["throttling_active"] is None
    assert message["soft_temp_limit_active"] is None
    assert message["runtime_condition_occurred"] is None
    # The processing metrics remain real numbers.
    assert message["window_max_chunk_processing_ms"] == 1.4
    assert message["window_min_processing_headroom_ms"] == 98.6


def test_negative_processing_headroom_survives_the_wire():
    fields = _runtime_status_fields(
        window_max_chunk_processing_ms=150.0,
        window_min_processing_headroom_ms=-50.0,
    )

    message = decode_message(encode_runtime_status(**fields))

    assert message["window_min_processing_headroom_ms"] == -50.0


def test_runtime_status_is_rejected_under_schema_version_1():
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        schema_version=1,
    )

    with pytest.raises(
        ProtocolError,
        match="not defined in schema version 1",
    ):
        decode_message(frame)


def test_v1_sample_chunk_and_prediction_remain_decodable():
    chunk_frame = _mutated_frame(
        encode_sample_chunk(_chunk(), "114"),
        schema_version=1,
    )
    prediction_frame = _mutated_frame(
        encode_prediction(_event(), "114"),
        schema_version=1,
    )

    chunk_message = decode_message(chunk_frame)
    prediction_message = decode_message(prediction_frame)

    assert chunk_message["schema_version"] == 1
    assert chunk_message["samples"] == CHUNK_SAMPLES
    assert prediction_message["schema_version"] == 1
    assert prediction_message["target_peak_index"] == 2358


def test_missing_runtime_status_timing_field_fails_clearly():
    message = json.loads(
        encode_runtime_status(**_runtime_status_fields()).decode("utf-8")
    )
    del message["window_max_chunk_processing_ms"]

    with pytest.raises(
        ProtocolError,
        match="missing required fields: window_max_chunk_processing_ms",
    ):
        decode_message(json.dumps(message))


@pytest.mark.parametrize("bad_value", [None, "1.4", True])
def test_non_numeric_timing_fields_fail_clearly(bad_value):
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        window_max_chunk_processing_ms=bad_value,
    )

    with pytest.raises(
        ProtocolError,
        match="'window_max_chunk_processing_ms' must be a number",
    ):
        decode_message(frame)


def test_schema_version_4_is_not_yet_supported():
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        schema_version=4,
    )

    with pytest.raises(ProtocolError, match="unsupported schema version 4"):
        decode_message(frame)


# ---------------------------------------------------------------------
#              Model-Stage Telemetry (Schema Version 3)
# ---------------------------------------------------------------------


def _v2_runtime_status_frame() -> str:
    """A v2 runtime_status exactly as a pre-v3 sender would emit it."""

    message = json.loads(
        encode_runtime_status(**_runtime_status_fields()).decode("utf-8")
    )
    message["schema_version"] = 2

    for field in (
        "model_inference_mean_ms",
        "model_throughput_sequences_per_second",
        "model_measurement_age_seconds",
    ):
        del message[field]

    return json.dumps(message)


def test_a_v2_runtime_status_without_model_fields_still_decodes():
    message = decode_message(_v2_runtime_status_frame())

    assert message["schema_version"] == 2
    assert message["window_max_chunk_processing_ms"] == 1.4
    assert "model_inference_mean_ms" not in message


def test_model_fields_may_all_be_null_before_the_first_inference():
    fields = _runtime_status_fields(
        model_inference_mean_ms=None,
        model_throughput_sequences_per_second=None,
        model_measurement_age_seconds=None,
    )

    message = decode_message(encode_runtime_status(**fields))

    assert message["model_inference_mean_ms"] is None
    assert message["model_throughput_sequences_per_second"] is None
    assert message["model_measurement_age_seconds"] is None


@pytest.mark.parametrize(
    "field",
    [
        "model_inference_mean_ms",
        "model_throughput_sequences_per_second",
        "model_measurement_age_seconds",
    ],
)
def test_missing_model_fields_fail_clearly_at_version_3(field):
    message = json.loads(
        encode_runtime_status(**_runtime_status_fields()).decode("utf-8")
    )
    del message[field]

    with pytest.raises(
        ProtocolError,
        match=f"missing required fields: {field}",
    ):
        decode_message(json.dumps(message))


@pytest.mark.parametrize(
    "field",
    [
        "model_inference_mean_ms",
        "model_throughput_sequences_per_second",
        "model_measurement_age_seconds",
    ],
)
@pytest.mark.parametrize("bad_value", [True, False, "1.4"])
def test_non_numeric_model_fields_fail_clearly(field, bad_value):
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        **{field: bad_value},
    )

    with pytest.raises(
        ProtocolError,
        match=f"'{field}' must be null or a number",
    ):
        decode_message(frame)


@pytest.mark.parametrize(
    "field",
    [
        "model_inference_mean_ms",
        "model_throughput_sequences_per_second",
    ],
)
@pytest.mark.parametrize("bad_value", [0, 0.0, -1.41, -709.2])
def test_non_positive_latency_and_throughput_fail_clearly(field, bad_value):
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        **{field: bad_value},
    )

    with pytest.raises(
        ProtocolError,
        match=f"'{field}' must be null or positive",
    ):
        decode_message(frame)


def test_a_negative_measurement_age_fails_clearly():
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        model_measurement_age_seconds=-0.5,
    )

    with pytest.raises(
        ProtocolError,
        match="'model_measurement_age_seconds' must be null or non-negative",
    ):
        decode_message(frame)


def test_a_zero_measurement_age_is_a_valid_fresh_measurement():
    message = decode_message(
        encode_runtime_status(
            **_runtime_status_fields(model_measurement_age_seconds=0.0)
        )
    )

    assert message["model_measurement_age_seconds"] == 0.0


def test_non_finite_model_values_fail_clearly():
    frame = _mutated_frame(
        encode_runtime_status(**_runtime_status_fields()),
        model_inference_mean_ms=float("inf"),
    )

    with pytest.raises(
        ProtocolError,
        match="'model_inference_mean_ms' must be finite",
    ):
        decode_message(frame)
