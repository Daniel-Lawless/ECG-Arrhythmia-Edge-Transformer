import threading
import time

import numpy as np

from ecg_arrhythmia.dashboard.state import (
    CONNECTION_CONNECTED,
    CONNECTION_DISCONNECTED,
    CONNECTION_LISTENING,
    DashboardState,
)
from ecg_arrhythmia.dashboard.stream_service import DashboardStreamService
from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.transport.tcp_sender import TCPStreamSender

# Bounded polling: only ever slow when a test is failing.
WAIT_TIMEOUT_SECONDS = 5.0


def _wait_until(condition, timeout=WAIT_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if condition():
            return True

        time.sleep(0.01)

    return False


def _chunk(start_index: int = 0) -> SampleChunk:
    return SampleChunk(
        samples=np.full(36, 0.125, dtype=np.float64),
        start_index=start_index,
        sampling_rate=360.0,
    )


def _event(target: int = 20) -> PredictionEvent:
    return PredictionEvent(
        target_peak_index=target,
        peak_indices=(10, target, 30),
        logits=np.array([3.5, -1.25, 0.5, -2.0], dtype=np.float32),
        predicted_class_index=0,
        predicted_label="N",
    )


def _started_service() -> tuple[DashboardState, DashboardStreamService]:
    state = DashboardState()
    service = DashboardStreamService(state, host="127.0.0.1", port=0)
    service.start()

    return state, service


# ---------------------------------------------------------------------
#                         Service Lifecycle
# ---------------------------------------------------------------------


def test_start_and_stop_terminate_the_background_thread_cleanly():
    state, service = _started_service()

    try:
        assert service.running
        assert state.snapshot().connection_status == CONNECTION_LISTENING
    finally:
        service.stop()

    assert not service.running
    assert state.snapshot().connection_status == CONNECTION_DISCONNECTED

    # stop() is idempotent.
    service.stop()
    assert not service.running


def test_start_is_idempotent_and_never_spawns_a_second_thread():
    state, service = _started_service()

    try:
        service.start()

        service_threads = [
            thread
            for thread in threading.enumerate()
            if thread.name == "dashboard-stream-service"
        ]

        assert len(service_threads) == 1
    finally:
        service.stop()


def test_the_service_can_be_restarted_after_a_stop():
    state, service = _started_service()
    service.stop()

    service.start()

    try:
        assert service.running
        assert state.snapshot().connection_status == CONNECTION_LISTENING
    finally:
        service.stop()


def test_stop_unblocks_a_thread_with_a_connected_client():
    state, service = _started_service()
    sender = TCPStreamSender(host="127.0.0.1", port=service.bound_port)
    sender.connect()

    try:
        sender.send_sample_chunk(_chunk(0), record_name="114")

        assert _wait_until(lambda: state.snapshot().chunks_received >= 1)

        # The receive thread is now blocked in recv(); stop() must
        # still return promptly.
        service.stop()

        assert not service.running
    finally:
        sender.close()


# ---------------------------------------------------------------------
#                      Localhost End To End
# ---------------------------------------------------------------------


def test_a_real_sender_stream_appears_in_the_snapshot():
    state, service = _started_service()

    try:
        with TCPStreamSender(host="127.0.0.1", port=service.bound_port) as sender:
            sender.send_sample_chunk(_chunk(0), record_name="114")
            sender.send_prediction(_event(20), record_name="114")

            assert _wait_until(lambda: state.snapshot().predictions_received >= 1)

            snapshot = state.snapshot()

            assert snapshot.connection_status == CONNECTION_CONNECTED
            assert snapshot.current_record_name == "114"
            assert snapshot.sampling_rate == 360.0
            assert snapshot.sample_indices == tuple(range(0, 36))
            assert snapshot.samples == (0.125,) * 36
            assert snapshot.chunks_received == 1

            visible = snapshot.visible_predictions
            assert len(visible) == 1
            assert visible[0].target_peak_index == 20
            assert visible[0].predicted_label == "N"
            assert visible[0].logits == (3.5, -1.25, 0.5, -2.0)

        # Clean client disconnect returns the service to listening.
        assert _wait_until(
            lambda: state.snapshot().connection_status == CONNECTION_LISTENING
        )
    finally:
        service.stop()


def test_a_runtime_status_reaches_the_snapshot_over_localhost():
    state, service = _started_service()

    try:
        with TCPStreamSender(host="127.0.0.1", port=service.bound_port) as sender:
            sender.send_sample_chunk(_chunk(0), record_name="114")
            sender.send_runtime_status(
                {
                    "record_name": "114",
                    "latest_sample_index": 35,
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
                    "model_inference_mean_ms": 1.41,
                    "model_throughput_sequences_per_second": 709.2,
                    "model_measurement_age_seconds": 0.0,
                }
            )

            assert _wait_until(lambda: state.snapshot().runtime_statuses_received >= 1)

            snapshot = state.snapshot()
            status = snapshot.runtime_status

            assert status.temperature_c == 48.7
            assert status.cpu_governor == "performance"
            assert status.throttling_active is False
            assert status.runtime_condition_active is False
            assert status.runtime_condition_occurred is False
            assert status.window_min_processing_headroom_ms == 98.6
            assert status.model_inference_mean_ms == 1.41
            assert status.model_throughput_sequences_per_second == 709.2
            assert snapshot.runtime_status_age_seconds is not None
            # The ECG chunk arrived alongside the telemetry.
            assert snapshot.chunks_received == 1
    finally:
        service.stop()


def test_a_second_client_is_accepted_and_the_view_is_reset():
    state, service = _started_service()

    try:
        with TCPStreamSender(host="127.0.0.1", port=service.bound_port) as sender:
            sender.send_sample_chunk(_chunk(0), record_name="114")

        assert _wait_until(lambda: state.snapshot().chunks_received == 1)
        assert _wait_until(
            lambda: state.snapshot().connection_status == CONNECTION_LISTENING
        )

        with TCPStreamSender(host="127.0.0.1", port=service.bound_port) as sender:
            sender.send_sample_chunk(_chunk(5000), record_name="122")

            assert _wait_until(lambda: state.snapshot().chunks_received == 2)

            snapshot = state.snapshot()

            # The new session's view contains only the new stream.
            assert snapshot.current_record_name == "122"
            assert snapshot.sample_indices == tuple(range(5000, 5036))
            # No discontinuity: the session reset cleared expectations.
            assert snapshot.discontinuities == 0
            # Lifetime counters remain cumulative across sessions.
            assert snapshot.chunks_received == 2
    finally:
        service.stop()
