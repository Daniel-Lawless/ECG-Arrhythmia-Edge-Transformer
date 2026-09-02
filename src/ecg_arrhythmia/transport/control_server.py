import argparse
import logging
import os
import socket
import threading
from pathlib import Path

from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplayMode
from ecg_arrhythmia.transport.control_config import DEFAULT_CONTROL_PORT
from ecg_arrhythmia.transport.control_protocol import (
    COMMAND_START_RECORD,
    COMMAND_STOP,
    DEFAULT_DEMO_RECORD,
    FRAME_DELIMITER,
    MAX_FRAME_BYTES,
    STATUS_ERROR,
    STATUS_OK,
    ControlProtocolError,
    decode_request,
    encode_response,
)
from ecg_arrhythmia.transport.protocol import TransportError
from ecg_arrhythmia.transport.send_record import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
    run_record_stream,
)
from ecg_arrhythmia.transport.tcp_receiver import DEFAULT_PORT as DEFAULT_DATA_PORT

logger = logging.getLogger(__name__)

# Bind on all interfaces: the agent must be reachable from the PC over
# the direct link, not only from the Pi itself.
DEFAULT_CONTROL_BIND_HOST = "0.0.0.0"

# DEFAULT_CONTROL_PORT is imported from control_config: the client must
# agree on it without importing this module, which would put the Pi's
# streaming dependencies on the dashboard machine.

# A control exchange is one short line each way; a client that sends
# nothing is dropped rather than holding the single-connection loop.
CONTROL_TIMEOUT_SECONDS = 10.0

# How long to wait for a running replay to notice a stop request. The
# flag is polled once per chunk (~100 ms in real-time mode), so this is
# generous; exceeding it means the worker is wedged somewhere else.
STREAM_STOP_TIMEOUT_SECONDS = 15.0


class RecordStreamRunner:
    """
    Own the at-most-one demo stream and its worker thread.

    Separated from the socket code so the start/stop/status state
    machine is testable without any networking, and so the streaming
    callable can be substituted in tests.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_DATA_PORT,
        model_path: Path = DEFAULT_MODEL_PATH,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        mode: str = ReplayMode.REAL_TIME.value,
        runtime_status_interval: float = DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
        stream_callable=run_record_stream,
    ) -> None:
        self.host = host
        self.port = port
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.mode = mode
        self.runtime_status_interval = runtime_status_interval
        self._stream_callable = stream_callable

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._record: str | None = None
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._is_running_locked()

    @property
    def current_record(self) -> str | None:
        """The record being streamed, or None when idle."""

        with self._lock:
            return self._record if self._is_running_locked() else None

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def _is_running_locked(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, record: str) -> str:
        """
        Accept a start request for `record`, replacing any running stream.

        Stopping first is what makes the dashboard's record dropdown
        behave as a user expects: choosing a new record switches to it
        rather than failing until the previous record ends. stop()
        joins the outgoing worker, so its data connection is already
        closed before the replacement worker is created - the PC
        receiver accepts one sender, and two overlapping workers would
        leave the second stalled in the TCP backlog.

        Returns an ACCEPTANCE message, not a completion one: this
        returns as soon as the worker thread exists, before the model
        session is built, the record is fetched or the data connection
        is established. Whether streaming actually began is visible in
        the dashboard's own connection state and telemetry, and any
        failure lands in last_error.
        """

        replaced = self.stop()

        with self._lock:
            self._stop_event = threading.Event()
            self._record = record
            self._last_error = None
            stop_event = self._stop_event

            thread = threading.Thread(
                target=self._run,
                args=(record, stop_event),
                name=f"record-stream-{record}",
                daemon=True,
            )
            self._thread = thread

        thread.start()
        logger.info(
            "Start accepted for record %s -> %s:%d",
            record,
            self.host,
            self.port,
        )

        if replaced is not None:
            return f"stopped record {replaced}; start accepted for record {record}"

        return f"start accepted for record {record}"

    def stop(self) -> str | None:
        """
        Stop any running stream; returns the record stopped, else None.

        The worker is asked to stop cooperatively at the next chunk
        boundary, so the stream always ends on a coherent message
        sequence and the socket is closed by the sender's own context
        manager.
        """

        with self._lock:
            if not self._is_running_locked():
                return None

            thread = self._thread
            record = self._record
            self._stop_event.set()

        if thread is not None:
            thread.join(timeout=STREAM_STOP_TIMEOUT_SECONDS)

            if thread.is_alive():
                logger.error(
                    "Record stream %s did not stop within %ss",
                    record,
                    STREAM_STOP_TIMEOUT_SECONDS,
                )
                # Keep the handle: a stop that failed must never look
                # like an idle agent, or the next start would create a
                # second concurrent sender.
                raise TransportError(
                    f"record {record} did not stop within "
                    f"{STREAM_STOP_TIMEOUT_SECONDS}s"
                )

        with self._lock:
            self._thread = None
            self._record = None

        logger.info("Stopped record %s", record)

        return record

    def _run(self, record: str, stop_event: threading.Event) -> None:
        try:
            summary = self._stream_callable(
                host=self.host,
                port=self.port,
                record=record,
                model_path=self.model_path,
                chunk_size=self.chunk_size,
                mode=self.mode,
                runtime_status_interval=self.runtime_status_interval,
                should_stop=stop_event.is_set,
            )
            logger.info("Record %s finished: %s", record, summary)
        except Exception as error:
            # The worker thread is the only place a streaming failure
            # (unreachable PC, missing record data) can surface, so it
            # is recorded for the next status request rather than lost.
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"

            logger.exception("Record stream %s failed", record)

    def status_message(self) -> str:
        # "worker running" rather than "streaming": the agent knows its
        # thread is alive, not that data is reaching the PC.
        record = self.current_record

        if record is not None:
            return f"stream worker running for record {record}"

        error = self.last_error

        if error is not None:
            return f"idle; last stream failed: {error}"

        return "idle"


class ControlServer:
    """
    Accept control connections and drive a RecordStreamRunner.

    One connection at a time, one request/response per connection: the
    exchange is tiny and rare, so a threaded server would add lifecycle
    complexity for no benefit.
    """

    def __init__(
        self,
        runner: RecordStreamRunner,
        host: str = DEFAULT_CONTROL_BIND_HOST,
        port: int = DEFAULT_CONTROL_PORT,
    ) -> None:
        self.runner = runner
        self.host = host
        self.port = port
        self._listener: socket.socket | None = None
        self._stop_event = threading.Event()

    def listen(self) -> None:
        if self._listener is not None:
            raise TransportError("control server is already listening")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            listener.bind((self.host, self.port))
        except OSError as error:
            listener.close()
            raise TransportError(
                f"could not bind control port {self.host}:{self.port}: {error}"
            ) from error

        listener.listen(1)
        self._listener = listener
        logger.info("Control server listening on %s:%d", self.host, self.bound_port)

    @property
    def bound_port(self) -> int:
        if self._listener is None:
            raise TransportError("control server is not listening")

        return self._listener.getsockname()[1]

    def serve_forever(self) -> None:
        """Handle control connections until close() is called."""

        if self._listener is None:
            raise TransportError("control server is not listening; call listen()")

        while not self._stop_event.is_set():
            try:
                connection, address = self._listener.accept()
            except OSError:
                if self._stop_event.is_set():
                    return

                raise

            with connection:
                try:
                    self._handle_connection(connection, address)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Control connection handling failed")

    def serve_once(self) -> None:
        """Handle exactly one control connection (used by tests)."""

        if self._listener is None:
            raise TransportError("control server is not listening; call listen()")

        connection, address = self._listener.accept()

        with connection:
            self._handle_connection(connection, address)

    def _handle_connection(self, connection: socket.socket, address: tuple) -> None:
        connection.settimeout(CONTROL_TIMEOUT_SECONDS)
        buffer = bytearray()

        while FRAME_DELIMITER not in buffer:
            try:
                data = connection.recv(1024)
            except OSError as error:
                logger.warning("Control receive from %s failed: %s", address, error)

                return

            if not data:
                logger.debug("Control client %s closed without a request", address)

                return

            buffer.extend(data)

            if len(buffer) > MAX_FRAME_BYTES:
                connection.sendall(
                    encode_response(
                        STATUS_ERROR,
                        "control request too large",
                        running=self.runner.running,
                        record=self.runner.current_record,
                    )
                )

                return

        frame = bytes(buffer).split(FRAME_DELIMITER, 1)[0]
        connection.sendall(self._respond(frame))

    def _respond(self, frame: bytes) -> bytes:
        try:
            request = decode_request(frame)
        except ControlProtocolError as error:
            # Invalid input is a normal event on a network listener,
            # not an agent failure: answer and keep serving.
            logger.warning("Rejected control request: %s", error)

            return encode_response(
                STATUS_ERROR,
                str(error),
                running=self.runner.running,
                record=self.runner.current_record,
            )

        command = request["command"]

        try:
            if command == COMMAND_START_RECORD:
                message = self.runner.start(request["record"])
            elif command == COMMAND_STOP:
                stopped = self.runner.stop()
                message = (
                    f"stopped record {stopped}"
                    if stopped is not None
                    else "no stream was running"
                )
            else:
                message = self.runner.status_message()
        except Exception as error:
            logger.exception("Control command %s failed", command)

            return encode_response(
                STATUS_ERROR,
                f"{type(error).__name__}: {error}",
                running=self.runner.running,
                record=self.runner.current_record,
            )

        return encode_response(
            STATUS_OK,
            message,
            running=self.runner.running,
            record=self.runner.current_record,
        )

    def close(self) -> None:
        """Stop serving and release the port. Safe to call twice."""

        self._stop_event.set()
        listener = self._listener

        if listener is None:
            return

        try:
            listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        listener.close()
        self._listener = None
        logger.info("Control server closed")

    def __enter__(self) -> "ControlServer":
        self.listen()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI values override container environment defaults; native CLI is unchanged."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Pi control agent so the dashboard can start demo "
            "records without SSH (Section 6.5)."
        ),
        epilog=(
            "Two separate connections are involved, in opposite "
            "directions:\n"
            "\n"
            "  OUTBOUND ECG DATA   Pi --> PC     --host/--port\n"
            "      the PC's dashboard receiver, e.g. 192.168.137.1:8765.\n"
            "      Set once here; never taken from a control request, so\n"
            "      no control client can redirect the ECG stream.\n"
            "\n"
            "  INBOUND CONTROL     PC --> Pi     --control-host/"
            "--control-port\n"
            "      the address THIS agent listens on for dashboard\n"
            "      commands, by default 0.0.0.0:8767 (all interfaces).\n"
            "\n"
            "Typical use on the Pi:\n"
            "  python -m ecg_arrhythmia.transport.control_server \\\n"
            "      --host 192.168.137.1 --port 8765\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("ECG_DATA_HOST"),
        metavar="PC_DATA_ADDRESS",
        help=(
            "OUTBOUND: address of the PC dashboard receiver that ECG "
            "data is streamed TO (the PC's Ethernet IP, e.g. "
            "192.168.137.1). Defaults to ECG_DATA_HOST; one must be supplied. "
            "Not the address this agent listens on."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("ECG_DATA_PORT", str(DEFAULT_DATA_PORT)),
        metavar="PC_DATA_PORT",
        help=(
            f"OUTBOUND: port of the PC dashboard data receiver "
            f"(ECG_DATA_PORT, otherwise {DEFAULT_DATA_PORT})."
        ),
    )
    parser.add_argument(
        "--control-host",
        type=str,
        default=DEFAULT_CONTROL_BIND_HOST,
        metavar="PI_BIND_ADDRESS",
        help=(
            f"INBOUND: address on this Pi to listen for dashboard "
            f"commands (default {DEFAULT_CONTROL_BIND_HOST}, all "
            f"interfaces)."
        ),
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        metavar="PI_CONTROL_PORT",
        help=(
            f"INBOUND: port on this Pi for dashboard commands "
            f"(default {DEFAULT_CONTROL_PORT})."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in ReplayMode],
        default=ReplayMode.REAL_TIME.value,
    )
    parser.add_argument(
        "--runtime-status-interval",
        type=float,
        default=DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--start-record",
        type=str,
        default=None,
        help=(
            f"Optionally begin streaming this record immediately "
            f"(for example {DEFAULT_DEMO_RECORD})."
        ),
    )

    args = parser.parse_args(argv)

    if args.host is None or not args.host.strip():
        parser.error("provide --host or set ECG_DATA_HOST to the PC data address")

    if not 1 <= args.port <= 65535:
        parser.error("--port / ECG_DATA_PORT must be between 1 and 65535")

    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    runner = RecordStreamRunner(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        mode=args.mode,
        runtime_status_interval=args.runtime_status_interval,
    )
    server = ControlServer(
        runner,
        host=args.control_host,
        port=args.control_port,
    )

    try:
        with server:
            if args.start_record is not None:
                runner.start(args.start_record)

            logger.info(
                "ECG data will be streamed to the PC at %s:%d; listening "
                "for dashboard commands on %s:%d",
                args.host,
                args.port,
                args.control_host,
                server.bound_port,
            )
            server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Control agent interrupted")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
