import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ecg_arrhythmia.dashboard.presentation import (
    condition_text,
    latest_prediction,
    recent_beats,
    record_time_seconds,
    rhythm_from_prediction,
    stable_softmax,
)
from ecg_arrhythmia.dashboard.state import DashboardState

logger = logging.getLogger(__name__)

DEFAULT_LIVE_ECG_HOST = "127.0.0.1"
DEFAULT_LIVE_ECG_PORT = 8766

# The recent-beat strip length shared with the browser renderer.
RECENT_BEAT_LIMIT = 12

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")

_plotly_js_cache: bytes | None = None


def allowed_origin(origin: str | None) -> str | None:
    """
    The Origin value to reflect, or None to send no CORS header.

    Only loopback origins are allowed (any port, so a non-default
    Streamlit port keeps working); anything else - or a missing or
    unparseable Origin - gets no Access-Control-Allow-Origin at all.
    """

    if not origin:
        return None

    parsed = urlparse(origin)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.hostname not in _LOOPBACK_HOSTS:
        return None

    return origin


def _prediction_entry(prediction, sampling_rate) -> dict:
    """One independently renderable prediction event for the browser."""

    scores = stable_softmax(prediction.logits)

    return {
        "target_peak_index": prediction.target_peak_index,
        "predicted_class_index": prediction.predicted_class_index,
        "predicted_label": prediction.predicted_label,
        "time_seconds": record_time_seconds(
            prediction.target_peak_index,
            sampling_rate,
        ),
        "scores": list(scores) if scores is not None else None,
    }


def build_live_payload(snapshot) -> dict:
    """
    The complete live-dashboard JSON for ONE immutable snapshot.

    Atomicity is by construction: the caller passes a single snapshot
    and every section of the payload comes from it. Derived values
    (softmax scores, RR/HR, times, condition text) are computed with
    the tested presentation helpers so there is one source of truth.
    An empty snapshot serialises to a valid waiting-state payload
    rather than an error.
    """

    indices = snapshot.sample_indices
    samples = snapshot.samples

    if indices and indices[-1] - indices[0] + 1 != len(indices):
        raise ValueError(
            "snapshot sample indices are not contiguous; refusing to "
            "compress the waveform to start_index + samples"
        )

    latest = latest_prediction(snapshot.recent_predictions)
    rhythm = rhythm_from_prediction(latest, snapshot.sampling_rate)

    latest_payload = (
        _prediction_entry(latest, snapshot.sampling_rate)
        if latest is not None
        else None
    )

    status = snapshot.runtime_status
    runtime_payload = None

    if status is not None:
        runtime_payload = {
            "temperature_c": status.temperature_c,
            "process_cpu_percent": status.process_cpu_percent,
            "process_rss_mib": status.process_rss_mib,
            "available_ram_mib": status.available_ram_mib,
            "cpu_frequency_mhz": status.cpu_frequency_mhz,
            "cpu_governor": status.cpu_governor,
            "window_max_chunk_processing_ms": (status.window_max_chunk_processing_ms),
            "window_min_processing_headroom_ms": (
                status.window_min_processing_headroom_ms
            ),
            "under_voltage_active": status.under_voltage_active,
            "frequency_capped_active": status.frequency_capped_active,
            "throttling_active": status.throttling_active,
            "soft_temp_limit_active": status.soft_temp_limit_active,
            "runtime_condition_active": status.runtime_condition_active,
            "runtime_condition_text": condition_text(status.runtime_condition_active),
            "runtime_condition_occurred": status.runtime_condition_occurred,
            # Live model-stage measurements (null before the first
            # inference or from a pre-v3 sender; same snapshot as the
            # rest of the runtime telemetry).
            "model_inference_mean_ms": status.model_inference_mean_ms,
            "model_throughput_sequences_per_second": (
                status.model_throughput_sequences_per_second
            ),
            "model_measurement_age_seconds": status.model_measurement_age_seconds,
        }

    return {
        "connection_status": snapshot.connection_status,
        "connected": snapshot.connected,
        "record_name": snapshot.current_record_name,
        "stream_age_seconds": snapshot.last_message_age_seconds,
        "discontinuities": snapshot.discontinuities,
        "sampling_rate": snapshot.sampling_rate,
        "ecg": {
            "start_index": indices[0] if indices else None,
            "samples": list(samples),
            "latest_sample_index": snapshot.latest_sample_index,
        },
        "visible_predictions": [
            {
                "target_peak_index": prediction.target_peak_index,
                "predicted_label": prediction.predicted_label,
            }
            for prediction in snapshot.visible_predictions
        ],
        "latest_prediction": latest_payload,
        "latest_rr_seconds": rhythm.rr_seconds,
        "estimated_hr_bpm": rhythm.hr_bpm,
        # Chronological and each entry independently renderable
        # (label, class index, time, four scores): the browser's
        # sequential presentation queue replays these one at a time
        # without recomputing any semantics in JavaScript.
        "recent_beats": [
            _prediction_entry(prediction, snapshot.sampling_rate)
            for prediction in recent_beats(
                snapshot.recent_predictions,
                limit=RECENT_BEAT_LIMIT,
            )
        ],
        "runtime_status": runtime_payload,
        "runtime_status_age_seconds": snapshot.runtime_status_age_seconds,
    }


def _plotly_js_bytes() -> bytes:
    """The Plotly bundle from the installed plotly package (lazy)."""

    global _plotly_js_cache

    if _plotly_js_cache is None:
        from plotly.offline import get_plotlyjs

        _plotly_js_cache = get_plotlyjs().encode("utf-8")

    return _plotly_js_cache


def _make_handler(state: DashboardState):
    class LiveDashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API name
            if self.path == "/live":
                try:
                    payload = build_live_payload(state.snapshot())
                except Exception as error:
                    logger.exception("live dashboard payload build failed")
                    self._send(
                        500,
                        "application/json",
                        json.dumps({"error": str(error)}).encode("utf-8"),
                    )

                    return

                self._send(
                    200,
                    "application/json",
                    json.dumps(payload).encode("utf-8"),
                )
            elif self.path == "/plotly.js":
                self._send(
                    200,
                    "application/javascript",
                    _plotly_js_bytes(),
                    cache="public, max-age=86400",
                )
            else:
                self._send(
                    404,
                    "application/json",
                    b'{"error": "not found"}',
                )

        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes,
            cache: str = "no-store",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)

            origin = allowed_origin(self.headers.get("Origin"))

            if origin is not None:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # ~10 requests/s: keep them out of the console.
            logger.debug("live-dashboard http: " + format, *args)

    return LiveDashboardHandler


class LiveEcgServer:
    """
    Read-only localhost HTTP server over the existing DashboardState.

    Lifecycle mirrors DashboardStreamService: constructed and started
    once per Streamlit process behind st.cache_resource; start() is
    idempotent; stop() shuts the server down and joins its thread.
    serve_forever runs on a daemon thread with daemon request threads,
    so a forgotten server can never keep Python alive. Multiple
    browser tabs may poll the same instance.
    """

    def __init__(
        self,
        state: DashboardState,
        host: str = DEFAULT_LIVE_ECG_HOST,
        port: int = DEFAULT_LIVE_ECG_PORT,
    ) -> None:
        self.host = host
        self._server = ThreadingHTTPServer((host, port), _make_handler(state))
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def bound_port(self) -> int:
        """The actual port bound, resolving an ephemeral port request."""

        return self._server.server_address[1]

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="live-dashboard-http",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Live dashboard endpoint listening on %s:%d",
            self.host,
            self.bound_port,
        )

    def stop(self, join_timeout: float = 5.0) -> None:
        """
        Idempotent shutdown.

        shutdown() blocks forever unless serve_forever is running, so
        it is only called while the serving thread is alive;
        server_close() releases the bound socket either way.
        """

        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=join_timeout)
            self._thread = None

        self._server.server_close()
        logger.info("Live dashboard endpoint stopped")
