import argparse
import json
import logging
import socket
from collections.abc import Callable, Iterator

from ecg_arrhythmia.transport.protocol import (
    FRAME_DELIMITER,
    MESSAGE_TYPE_RUNTIME_STATUS,
    MESSAGE_TYPE_SAMPLE_CHUNK,
    ProtocolError,
    TransportError,
    decode_message,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

RECEIVE_BUFFER_BYTES = 4096

# A frame far larger than any legal message means the peer is not
# speaking this protocol (or newlines were lost); failing beats
# buffering a malformed stream without bound.
MAX_FRAME_BYTES = 1_048_576


class FrameAssembler:
    """
    Reassemble newline-delimited frames from arbitrary byte chunks.

    Pure buffering, no sockets, so framing behaviour is unit-testable
    on its own.
    """

    def __init__(self, max_frame_bytes: int = MAX_FRAME_BYTES) -> None:
        self._buffer = bytearray()
        self._max_frame_bytes = max_frame_bytes

    @property
    def pending_bytes(self) -> int:
        """Bytes buffered that do not yet form a complete frame."""

        return len(self._buffer)

    def feed(self, data: bytes) -> list[bytes]:
        """
        Add received bytes and return every completed frame, in order.

        Frames are returned without their newline delimiter.
        """

        self._buffer.extend(data)
        frames = []

        while True:
            delimiter = self._buffer.find(FRAME_DELIMITER)

            if delimiter == -1:
                break

            frames.append(bytes(self._buffer[:delimiter]))
            del self._buffer[: delimiter + 1]

        if len(self._buffer) > self._max_frame_bytes:
            raise ProtocolError(
                f"frame exceeds {self._max_frame_bytes} bytes without a "
                f"newline; peer is not speaking this protocol"
            )

        return frames


class TCPStreamReceiver:
    """
    Accept one sender connection and yield decoded messages.

    Lifecycle: listen() -> iterate messages() -> close(), or use the
    instance as a context manager. Bind with port 0 to receive an
    ephemeral port (read it back from bound_port), which keeps tests
    free of port collisions.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float | None = None,
        on_client_connected: Callable[[tuple], None] | None = None,
        on_client_disconnected: Callable[[], None] | None = None,
    ) -> None:
        # The optional lifecycle callbacks let a consumer (the Section
        # 6.2 dashboard service) track connection state explicitly
        # instead of inferring it from message arrival. They are invoked
        # on the thread that iterates messages().
        self.host = host
        self.port = port
        self.timeout = timeout
        self._on_client_connected = on_client_connected
        self._on_client_disconnected = on_client_disconnected
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None

    def listen(self) -> None:
        """Bind and listen for exactly one sender."""

        if self._listener is not None:
            raise TransportError("receiver is already listening")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            listener.bind((self.host, self.port))
        except OSError as error:
            listener.close()
            raise TransportError(
                f"could not bind to {self.host}:{self.port}: {error}"
            ) from error

        listener.listen(1)
        listener.settimeout(self.timeout)
        self._listener = listener
        logger.info("Receiver listening on %s:%d", self.host, self.bound_port)

    @property
    def bound_port(self) -> int:
        """The actual port bound, resolving an ephemeral port request."""

        if self._listener is None:
            raise TransportError("receiver is not listening; call listen() first")

        return self._listener.getsockname()[1]

    def messages(self) -> Iterator[dict]:
        """
        Accept one client and yield validated messages until clean EOF.

        Handles any recv() fragmentation: several frames per receive,
        or one frame spread across many receives. A disconnect with a
        partial frame buffered raises ProtocolError.
        """

        if self._listener is None:
            raise TransportError("receiver is not listening; call listen() first")

        try:
            connection, address = self._listener.accept()
        except TimeoutError as error:
            raise TransportError("timed out waiting for a client connection") from error
        except OSError as error:
            raise TransportError(f"accept failed: {error}") from error

        connection.settimeout(self.timeout)
        self._connection = connection
        logger.info("Client connected: %s:%d", address[0], address[1])

        if self._on_client_connected is not None:
            self._on_client_connected(address)

        assembler = FrameAssembler()

        try:
            while True:
                try:
                    data = connection.recv(RECEIVE_BUFFER_BYTES)
                except OSError as error:
                    raise TransportError(f"receive failed: {error}") from error

                if not data:
                    if assembler.pending_bytes:
                        raise ProtocolError(
                            f"connection closed mid-frame with "
                            f"{assembler.pending_bytes} unframed bytes buffered"
                        )

                    logger.info("Client disconnected cleanly")

                    return

                for frame in assembler.feed(data):
                    yield decode_message(frame)
        finally:
            connection.close()
            self._connection = None

            if self._on_client_disconnected is not None:
                self._on_client_disconnected()

    def close(self) -> None:
        """
        Stop listening and drop any active connection.

        Each socket is shut down before it is closed. This matters for
        cross-thread shutdown: on Linux, close() alone does NOT wake a
        thread blocked in accept() or recv() (the syscall keeps waiting
        on the old descriptor), whereas shutdown() interrupts accept()
        with an error and recv() with EOF. On Windows, shutdown() on a
        never-connected listener fails (ignored) and close() itself
        aborts the accept, so the sequence is correct on both
        platforms. Safe to call when already closed.
        """

        connection = self._connection

        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                connection.close()
            except OSError:
                pass

        listener = self._listener

        if listener is None:
            return

        try:
            listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        listener.close()
        self._listener = None
        logger.info("Receiver closed")

    def __enter__(self) -> "TCPStreamReceiver":
        self.listen()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def summarise_message(message: dict) -> str:
    """One console line per message so the CLI never floods the terminal."""

    if message["message_type"] == MESSAGE_TYPE_SAMPLE_CHUNK:
        return (
            f"sample_chunk record={message['record_name']} "
            f"start={message['start_index']} "
            f"samples={len(message['samples'])}"
        )

    if message["message_type"] == MESSAGE_TYPE_RUNTIME_STATUS:
        return (
            f"runtime_status record={message['record_name']} "
            f"temp={message['temperature_c']} "
            f"governor={message['cpu_governor']} "
            f"processing_max={message['window_max_chunk_processing_ms']}ms "
            f"headroom_min={message['window_min_processing_headroom_ms']}ms"
        )

    return (
        f"prediction record={message['record_name']} "
        f"target={message['target_peak_index']} "
        f"class={message['predicted_label']}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Listen for one ECG stream sender and print received messages "
            "(Section 6.1 verification receiver)."
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
        "--verbose",
        action="store_true",
        help="Print full message payloads instead of one-line summaries.",
    )

    args = parser.parse_args()

    try:
        with TCPStreamReceiver(host=args.host, port=args.port) as receiver:
            for message in receiver.messages():
                if args.verbose:
                    print(json.dumps(message))
                else:
                    print(summarise_message(message))
    except KeyboardInterrupt:
        logger.info("Receiver interrupted")
    except (ProtocolError, TransportError) as error:
        raise SystemExit(f"receiver error: {error}") from error


if __name__ == "__main__":
    main()
