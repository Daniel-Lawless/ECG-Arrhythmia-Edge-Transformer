import logging
import socket

from ecg_arrhythmia.transport.protocol import (
    TransportError,
    encode_prediction,
    encode_runtime_status,
    encode_sample_chunk,
)

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0


class TCPStreamSender:
    """
    Send wire frames to one receiver over a persistent TCP connection.

    Lifecycle: connect() -> send_*() calls -> close(), or use the
    instance as a context manager for the same sequence.
    """

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        send_timeout: float | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.send_timeout = send_timeout
        self._socket: socket.socket | None = None

    @property
    def connected(self) -> bool:
        """Whether a socket is currently held (not a liveness probe)."""

        return self._socket is not None

    def connect(self) -> None:
        """Open the persistent connection."""

        if self._socket is not None:
            raise TransportError("sender is already connected")

        try:
            self._socket = socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout,
            )
        except OSError as error:
            raise TransportError(
                f"could not connect to {self.host}:{self.port}: {error}"
            ) from error

        self._socket.settimeout(self.send_timeout)
        logger.info("Sender connected to %s:%d", self.host, self.port)

    def reconnect(self) -> None:
        """Close any existing socket and connect again."""

        self.close()
        self.connect()

    def send_sample_chunk(self, chunk, record_name: str | None = None) -> None:
        """Send one ECG chunk frame."""

        self._send(encode_sample_chunk(chunk, record_name))

    def send_prediction(self, event, record_name: str | None = None) -> None:
        """Send one prediction frame."""

        self._send(encode_prediction(event, record_name))

    def send_runtime_status(self, status: dict) -> None:
        """Send one runtime_status frame; status maps the wire fields."""

        self._send(encode_runtime_status(**status))

    def _send(self, frame: bytes) -> None:
        if self._socket is None:
            raise TransportError("sender is not connected; call connect() first")

        try:
            self._socket.sendall(frame)
        except OSError as error:
            raise TransportError(
                f"send to {self.host}:{self.port} failed: {error}"
            ) from error

    def close(self) -> None:
        """Close the connection; safe to call when already closed."""

        if self._socket is None:
            return

        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            # The peer may already have gone; closing is what matters.
            pass

        self._socket.close()
        self._socket = None
        logger.info("Sender disconnected from %s:%d", self.host, self.port)

    def __enter__(self) -> "TCPStreamSender":
        self.connect()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
