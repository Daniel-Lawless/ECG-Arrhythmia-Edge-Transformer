import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from ecg_arrhythmia.dashboard.live_ecg_server import (
    LiveEcgServer,
    allowed_origin,
    build_live_payload,
    live_endpoint_url,
)
from ecg_arrhythmia.dashboard.presentation import stable_softmax
from ecg_arrhythmia.dashboard.state import DashboardState, DashboardStateConfig

# 10 s at 100 Hz -> capacity 1000 samples.
CONFIG = DashboardStateConfig(ecg_window_seconds=10.0)


def test_live_url_preserves_native_defaults_and_actual_bound_port(monkeypatch):
    monkeypatch.delenv("ECG_LIVE_HTTP_PUBLIC_URL", raising=False)

    assert live_endpoint_url("127.0.0.1", 8766) == "http://127.0.0.1:8766"
    assert live_endpoint_url("localhost", 15321) == "http://localhost:15321"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://127.0.0.1:8766", "http://127.0.0.1:8766"),
        ("  http://localhost:9876/  ", "http://localhost:9876"),
        ("https://example.test/ecg/", "https://example.test/ecg"),
    ],
)
def test_live_url_uses_public_override_not_bind_address(
    monkeypatch, configured, expected
):
    monkeypatch.setenv("ECG_LIVE_HTTP_PUBLIC_URL", configured)

    assert live_endpoint_url("0.0.0.0", 12345) == expected


@pytest.mark.parametrize(
    "configured",
    [
        "",
        "   ",
        "/live",
        "localhost:8766",
        "ftp://localhost:8766",
        "http://0.0.0.0:8766",
        "http://[::]:8766",
        "http://localhost:not-a-port",
        "http://localhost:65536",
        "http://localhost:0",
        "http://user:password@localhost:8766",
        "http://localhost:8766?token=x",
        "http://localhost:8766#fragment",
        "http://local host:8766",
        "http://[invalid",
    ],
)
def test_live_url_rejects_invalid_explicit_configuration(monkeypatch, configured):
    monkeypatch.setenv("ECG_LIVE_HTTP_PUBLIC_URL", configured)

    with pytest.raises(ValueError, match="ECG_LIVE_HTTP_PUBLIC_URL"):
        live_endpoint_url("127.0.0.1", 8766)


# Argmax 2 = V, consistent with the predicted label used below.
V_LOGITS = [0.5, 0.2, 4.0, 0.1]


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _chunk_message(start, length, record="114", rate=100.0):
    return {
        "schema_version": 2,
        "message_type": "sample_chunk",
        "record_name": record,
        "start_index": start,
        "sampling_rate": rate,
        # Amplitude equals absolute index, so mapping is verifiable.
        "samples": [float(index) for index in range(start, start + length)],
    }


def _prediction_message(target, record="114", label="V", logits=None):
    return {
        "schema_version": 2,
        "message_type": "prediction",
        "record_name": record,
        "target_peak_index": target,
        "peak_indices": [target - 50, target],
        "logits": list(logits) if logits is not None else list(V_LOGITS),
        "predicted_class_index": 2,
        "predicted_label": label,
    }


def _runtime_status_message(**overrides):
    message = {
        "schema_version": 2,
        "message_type": "runtime_status",
        "record_name": "114",
        "latest_sample_index": 1099,
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
    }
    message.update(overrides)

    return message


def _populated_state(clock=None) -> DashboardState:
    state = DashboardState(CONFIG, clock=clock or FakeClock(100.0))
    state.apply_message(_chunk_message(1000, 100))
    state.apply_message(_prediction_message(1050, label="V"))
    # Outside the visible window: excluded from markers, kept in
    # recent history.
    state.apply_message(_prediction_message(900, label="N"))
    state.apply_message(_runtime_status_message())

    return state


# ---------------------------------------------------------------------
#                        Live Payload Builder
# ---------------------------------------------------------------------


def test_the_live_payload_is_one_coherent_snapshot_view():
    payload = build_live_payload(_populated_state().snapshot())

    assert payload["connection_status"] == "disconnected"
    assert payload["connected"] is False
    assert payload["record_name"] == "114"
    assert payload["discontinuities"] == 0
    assert payload["sampling_rate"] == 100.0
    assert isinstance(payload["stream_age_seconds"], float)

    assert payload["ecg"]["start_index"] == 1000
    assert len(payload["ecg"]["samples"]) == 100
    assert payload["ecg"]["latest_sample_index"] == 1099

    # RR/HR come from the tested presentation helper: the latest
    # prediction (N at 900) has peaks (850, 900) -> RR 0.5 s, HR 120.
    assert payload["latest_rr_seconds"] == pytest.approx(0.5)
    assert payload["estimated_hr_bpm"] == pytest.approx(120.0)

    runtime = payload["runtime_status"]

    assert runtime["temperature_c"] == pytest.approx(48.7)
    assert runtime["cpu_governor"] == "performance"
    assert runtime["runtime_condition_active"] is False
    assert runtime["runtime_condition_text"] == "OK"
    assert runtime["throttling_active"] is False
    assert isinstance(payload["runtime_status_age_seconds"], float)
    # The populated state used a v2 message, so the model-stage
    # measurements serialize as null - never fabricated zeros.
    assert runtime["model_inference_mean_ms"] is None
    assert runtime["model_throughput_sequences_per_second"] is None
    assert runtime["model_measurement_age_seconds"] is None


def test_model_measurements_reach_the_payload_from_the_same_snapshot():
    state = DashboardState(CONFIG, clock=FakeClock(100.0))
    state.apply_message(_chunk_message(1000, 100))
    state.apply_message(
        _runtime_status_message(
            schema_version=3,
            model_inference_mean_ms=1.41,
            model_throughput_sequences_per_second=709.2,
            model_measurement_age_seconds=0.25,
        )
    )

    runtime = build_live_payload(state.snapshot())["runtime_status"]

    assert runtime["model_inference_mean_ms"] == pytest.approx(1.41)
    assert runtime["model_throughput_sequences_per_second"] == pytest.approx(709.2)
    assert runtime["model_measurement_age_seconds"] == pytest.approx(0.25)


def test_one_prediction_appears_coherently_in_every_section():
    # Spec: a single V prediction must simultaneously be the visible
    # marker, the latest classification, the newest recent beat, and
    # carry scores derived from its own logits.
    state = DashboardState(CONFIG, clock=FakeClock(100.0))
    state.apply_message(_chunk_message(1000, 100))
    state.apply_message(_prediction_message(1050, label="V"))

    payload = build_live_payload(state.snapshot())

    assert payload["visible_predictions"] == [
        {"target_peak_index": 1050, "predicted_label": "V"}
    ]

    latest = payload["latest_prediction"]

    assert latest["target_peak_index"] == 1050
    assert latest["predicted_label"] == "V"
    assert latest["predicted_class_index"] == 2
    assert latest["time_seconds"] == pytest.approx(10.5)
    assert latest["scores"] == pytest.approx(list(stable_softmax(V_LOGITS)))
    # The predicted class carries the highest score.
    assert max(latest["scores"]) == latest["scores"][2]

    newest_beat = payload["recent_beats"][-1]

    assert newest_beat["target_peak_index"] == 1050
    assert newest_beat["predicted_label"] == "V"
    assert newest_beat["scores"] == pytest.approx(latest["scores"])

    # Marker amplitude derivable from the same payload.
    offset = latest["target_peak_index"] - payload["ecg"]["start_index"]

    assert payload["ecg"]["samples"][offset] == pytest.approx(1050.0)


def test_a_burst_of_predictions_remains_independently_renderable():
    # Sequential presentation needs every burst event to stand alone:
    # chronological order preserved, never collapsed to the latest,
    # each with its own label, class index, record time and scores.
    state = DashboardState(CONFIG, clock=FakeClock(100.0))
    state.apply_message(_chunk_message(1000, 100))

    burst = [
        (1010, "N", [3.0, 0.1, 0.2, 0.1]),
        (1030, "N", [2.5, 0.3, 0.1, 0.2]),
        (1050, "V", [0.5, 0.2, 4.0, 0.1]),
        (1070, "S", [0.2, 3.5, 0.4, 0.1]),
    ]

    for target, label, logits in burst:
        state.apply_message(_prediction_message(target, label=label, logits=logits))

    beats = build_live_payload(state.snapshot())["recent_beats"]

    assert [beat["predicted_label"] for beat in beats] == ["N", "N", "V", "S"]
    assert [beat["target_peak_index"] for beat in beats] == [
        1010,
        1030,
        1050,
        1070,
    ]

    for (target, _, logits), beat in zip(burst, beats, strict=True):
        assert beat["time_seconds"] == pytest.approx(target / 100.0)
        assert beat["scores"] == pytest.approx(list(stable_softmax(logits)))
        assert sum(beat["scores"]) == pytest.approx(1.0)


def test_under_voltage_is_a_condition_warning_but_never_throttling():
    state = DashboardState(CONFIG, clock=FakeClock(100.0))
    state.apply_message(_runtime_status_message(under_voltage_active=True))

    runtime = build_live_payload(state.snapshot())["runtime_status"]

    assert runtime["under_voltage_active"] is True
    assert runtime["runtime_condition_active"] is True
    assert runtime["runtime_condition_text"] == "Warning"
    assert runtime["throttling_active"] is False


def test_an_empty_snapshot_serialises_to_a_valid_waiting_payload():
    payload = build_live_payload(DashboardState(CONFIG).snapshot())

    assert payload["connection_status"] == "disconnected"
    assert payload["record_name"] is None
    assert payload["stream_age_seconds"] is None
    assert payload["sampling_rate"] is None
    assert payload["ecg"] == {
        "start_index": None,
        "samples": [],
        "latest_sample_index": None,
    }
    assert payload["visible_predictions"] == []
    assert payload["latest_prediction"] is None
    assert payload["latest_rr_seconds"] is None
    assert payload["estimated_hr_bpm"] is None
    assert payload["recent_beats"] == []
    assert payload["runtime_status"] is None
    assert payload["runtime_status_age_seconds"] is None
    json.dumps(payload)


def test_the_payload_is_plain_json_encodable():
    payload = build_live_payload(_populated_state().snapshot())

    assert json.loads(json.dumps(payload)) == payload


def test_non_contiguous_sample_indices_are_rejected_not_flattened():
    broken = SimpleNamespace(
        connection_status="connected",
        connected=True,
        current_record_name="114",
        sampling_rate=100.0,
        sample_indices=(1000, 1001, 1005),
        samples=(0.1, 0.2, 0.3),
        visible_predictions=(),
        recent_predictions=(),
        latest_sample_index=1005,
        last_message_age_seconds=0.05,
        discontinuities=0,
        runtime_status=None,
        runtime_status_age_seconds=None,
    )

    with pytest.raises(ValueError, match="not contiguous"):
        build_live_payload(broken)


# ---------------------------------------------------------------------
#                            HTTP Server
# ---------------------------------------------------------------------


def _get(url: str, origin: str | None = None):
    request = urllib.request.Request(url)

    if origin is not None:
        request.add_header("Origin", origin)

    with urllib.request.urlopen(request, timeout=5.0) as response:
        return response.status, dict(response.headers), response.read()


def test_loopback_origins_are_reflected_and_others_get_no_cors_header():
    assert allowed_origin("http://localhost:8501") == "http://localhost:8501"
    assert allowed_origin("http://127.0.0.1:8502") == "http://127.0.0.1:8502"
    assert allowed_origin("http://192.168.137.20:8501") is None
    assert allowed_origin("https://evil.example") is None
    assert allowed_origin("file://x") is None
    assert allowed_origin(None) is None
    assert allowed_origin("") is None


def test_the_endpoint_serves_the_live_payload_with_the_designed_headers():
    server = LiveEcgServer(_populated_state(), host="127.0.0.1", port=0)
    server.start()

    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        status, headers, body = _get(
            f"{base}/live",
            origin="http://localhost:8501",
        )

        assert status == 200
        assert headers["Content-Type"] == "application/json"
        assert headers["Cache-Control"] == "no-store"
        # The loopback Streamlit origin is reflected, not wildcarded.
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:8501"
        assert headers["Vary"] == "Origin"

        payload = json.loads(body)

        assert payload["record_name"] == "114"
        assert payload["ecg"]["start_index"] == 1000
        assert payload["runtime_status"]["cpu_governor"] == "performance"

        # A non-loopback origin gets no CORS header at all.
        _, foreign_headers, _ = _get(
            f"{base}/live",
            origin="http://192.168.137.20:8501",
        )

        assert "Access-Control-Allow-Origin" not in foreign_headers

        # And a plain same-machine request without an Origin works.
        second_status, plain_headers, _ = _get(f"{base}/live")

        assert second_status == 200
        assert "Access-Control-Allow-Origin" not in plain_headers
    finally:
        server.stop()


def test_unknown_paths_including_the_old_route_return_404():
    server = LiveEcgServer(DashboardState(CONFIG), host="127.0.0.1", port=0)
    server.start()

    try:
        base = f"http://127.0.0.1:{server.bound_port}"

        for path in ("/other", "/ecg"):
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                _get(f"{base}{path}")

            assert excinfo.value.code == 404
    finally:
        server.stop()


def test_the_endpoint_serves_a_valid_empty_state_before_the_pi_connects():
    server = LiveEcgServer(DashboardState(CONFIG), host="127.0.0.1", port=0)
    server.start()

    try:
        status, _, body = _get(f"http://127.0.0.1:{server.bound_port}/live")

        assert status == 200

        payload = json.loads(body)

        assert payload["ecg"]["samples"] == []
        assert payload["latest_prediction"] is None
    finally:
        server.stop()


def test_stop_terminates_the_server_thread_cleanly_and_is_idempotent():
    server = LiveEcgServer(DashboardState(CONFIG), host="127.0.0.1", port=0)
    server.start()
    thread = server._thread

    server.stop()

    assert thread is not None
    assert not thread.is_alive()

    # A second stop must return promptly rather than blocking in
    # shutdown() with no serve_forever loop to acknowledge it.
    server.stop()
