import argparse
import logging
import threading
import time

from ecg_arrhythmia.dashboard.state import (
    DashboardState,
    DashboardStateConfig,
)
from ecg_arrhythmia.transport.protocol import ProtocolError, TransportError
from ecg_arrhythmia.transport.tcp_receiver import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    TCPStreamReceiver,
)

logger = logging.getLogger(__name__)

DEFAULT_STOP_JOIN_TIMEOUT_SECONDS = 5.0


class DashboardStreamService:
    """
    Run the TCP receiver in the background and feed a DashboardState.

    Lifecycle: start() binds the listener and launches the receive
    thread; state.snapshot() may be read at any time from any thread;
    stop() shuts the sockets, joins the thread and marks the state
    disconnected.
    """

    def __init__(
        self,
        state: DashboardState,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.state = state
        self.host = host
        self.port = port
        self._receiver: TCPStreamReceiver | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def bound_port(self) -> int:
        """The listening port (resolves an ephemeral port request)."""

        if self._receiver is None:
            raise TransportError("service is not started")

        return self._receiver.bound_port

    def start(self) -> None:
        """
        Bind, listen and launch the background receive thread.

        Idempotent: calling start() on a running service is a no-op, so
        an accidental repeated call cannot spawn a second thread. Bind
        failures raise TransportError synchronously.
        """

        if self.running:
            logger.info("Dashboard stream service already running")

            return

        self._stop_event.clear()
        self._receiver = TCPStreamReceiver(
            host=self.host,
            port=self.port,
            on_client_connected=self._handle_client_connected,
            on_client_disconnected=self._handle_client_disconnected,
        )
        self._receiver.listen()
        self.state.mark_listening()

        self._thread = threading.Thread(
            target=self._run,
            name="dashboard-stream-service",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Dashboard stream service started on %s:%d",
            self.host,
            self.bound_port,
        )

    def stop(self, join_timeout: float = DEFAULT_STOP_JOIN_TIMEOUT_SECONDS) -> None:
        """
        Stop the receive thread promptly and mark the state stopped.

        Closing the receiver's sockets unblocks accept()/recv(), so the
        thread exits without forceful termination. Idempotent.
        """

        self._stop_event.set()

        if self._receiver is not None:
            self._receiver.close()

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

            if self._thread.is_alive():
                # Keep the handle so running stays True: a stop that
                # failed must never masquerade as a stopped service.
                logger.error("Receive thread did not stop within %ss", join_timeout)
            else:
                self._thread = None

        self._receiver = None
        self.state.mark_stopped()
        logger.info("Dashboard stream service stopped")

    def _handle_client_connected(self, address: tuple) -> None:
        # Runs on the receive thread via the receiver callback.
        self.state.mark_client_connected()

    def _handle_client_disconnected(self) -> None:
        if not self._stop_event.is_set():
            self.state.mark_client_disconnected()

    def _run(self) -> None:
        # A local reference: stop() nulls self._receiver, and the
        # receive thread must never race that assignment.
        receiver = self._receiver

        if receiver is None:
            return

        try:
            while not self._stop_event.is_set():
                try:
                    for message in receiver.messages():
                        self.state.apply_message(message)
                except (ProtocolError, TransportError) as error:
                    if self._stop_event.is_set():
                        return

                    # One bad session must not kill the dashboard:
                    # record it and go back to listening for the next
                    # client. The disconnect callback has already reset
                    # the connection status.
                    self.state.record_error(str(error))
                    logger.warning("Stream session ended with error: %s", error)
        except Exception as error:  # pragma: no cover - defensive
            # An unexpected programming error must be visible, not
            # silently swallowed by the background thread.
            self.state.record_error(f"receive loop terminated unexpectedly: {error}")
            logger.exception("Dashboard stream service receive loop crashed")


def _optional_text(value, format_spec: str, suffix: str = "") -> str:
    if value is None:
        return "n/a"

    return f"{value:{format_spec}}{suffix}"


def _snapshot_line(snapshot) -> str:
    line = (
        f"status={snapshot.connection_status} "
        f"record={snapshot.current_record_name} "
        f"samples={len(snapshot.samples)} "
        f"latest_sample={snapshot.latest_sample_index} "
        f"predictions={snapshot.predictions_received} "
        f"discontinuities={snapshot.discontinuities} "
        f"age={_optional_text(snapshot.last_message_age_seconds, '.2f', 's')}"
    )

    status = snapshot.runtime_status

    if status is not None:
        line += (
            f" temp={_optional_text(status.temperature_c, '.1f', 'C')}"
            f" cpu={_optional_text(status.process_cpu_percent, '.1f', '%')}"
            f" rss={_optional_text(status.process_rss_mib, '.0f', 'MiB')}"
            f" freq={_optional_text(status.cpu_frequency_mhz, '.0f', 'MHz')}"
            f" governor={status.cpu_governor}"
            f" throttling={status.throttling_active}"
            f" condition_active={status.runtime_condition_active}"
            f" condition_occurred={status.runtime_condition_occurred}"
            f" processing_max={status.window_max_chunk_processing_ms:.1f}ms"
            f" headroom_min={status.window_min_processing_headroom_ms:.1f}ms"
            f" runtime_age="
            f"{_optional_text(snapshot.runtime_status_age_seconds, '.2f', 's')}"
        )

    return line


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Run the dashboard stream service and print one state "
            "snapshot per interval (Section 6.2 verification CLI)."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Bind address; use 0.0.0.0 explicitly to accept LAN clients.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between snapshot lines.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DashboardStateConfig().ecg_window_seconds,
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=DashboardStateConfig().max_prediction_history,
    )

    args = parser.parse_args()

    state = DashboardState(
        DashboardStateConfig(
            ecg_window_seconds=args.window_seconds,
            max_prediction_history=args.max_predictions,
        )
    )
    service = DashboardStreamService(state, host=args.host, port=args.port)
    service.start()

    try:
        while True:
            time.sleep(args.interval)
            print(_snapshot_line(state.snapshot()))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
