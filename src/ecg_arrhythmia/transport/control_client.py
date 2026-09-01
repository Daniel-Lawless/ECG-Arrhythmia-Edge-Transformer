import logging
import socket

from ecg_arrhythmia.transport.control_config import DEFAULT_CONTROL_PORT
from ecg_arrhythmia.transport.control_protocol import (
    COMMAND_START_RECORD,
    COMMAND_STATUS,
    COMMAND_STOP,
    FRAME_DELIMITER,
    MAX_FRAME_BYTES,
    ControlProtocolError,
    decode_response,
    encode_request,
)

logger = logging.getLogger(__name__)

# The Pi's address on the direct Pi<->PC Ethernet link used throughout
# this project. Overridable via the dashboard's ECG_PI_CONTROL_HOST
# environment variable so the address is configuration, not a value
# scattered through the code.
DEFAULT_PI_CONTROL_HOST = "192.168.137.27"

# Connecting is either immediate on the direct link or never: a short
# limit makes a wrong address or a stopped agent fail a UI click
# quickly instead of hanging it.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0

# Waiting for the reply is a different question. A start request that
# replaces a running stream first joins the outgoing worker, which the
# agent allows up to STREAM_STOP_TIMEOUT_SECONDS (15 s) to finish -
# normally milliseconds, but longer if the sender is blocked writing to
# a stalled receiver. Reading must therefore outlast the agent's own
# limit, or the dashboard would report a timeout for a request the Pi
# went on to complete successfully.
DEFAULT_READ_TIMEOUT_SECONDS = 20.0


class ControlClientError(RuntimeError):
    """A control request could not be completed."""


def send_command(
    command: str,
    record: str | None = None,
    host: str = DEFAULT_PI_CONTROL_HOST,
    port: int = DEFAULT_CONTROL_PORT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
) -> dict:
    """
    Send one control command and return the decoded response.

    Raises ControlClientError for unreachable agents, timeouts, closed
    connections and malformed replies, so every caller-visible failure
    is one exception type carrying a message fit to show a user.
    Invalid records raise ControlProtocolError from encode_request
    before any socket is opened.
    """

    request = encode_request(command, record)

    try:
        with socket.create_connection(
            (host, port),
            timeout=connect_timeout,
        ) as connection:
            connection.settimeout(read_timeout)
            connection.sendall(request)
            frame = _read_frame(connection)
    except TimeoutError as error:
        raise ControlClientError(
            f"timed out talking to the Pi control agent at {host}:{port}"
        ) from error
    except OSError as error:
        raise ControlClientError(
            f"could not reach the Pi control agent at {host}:{port}: {error}"
        ) from error

    try:
        return decode_response(frame)
    except ControlProtocolError as error:
        raise ControlClientError(
            f"the Pi control agent sent an unreadable response: {error}"
        ) from error


def start_record(record: str, **kwargs) -> dict:
    """Ask the agent to stream `record`, replacing any running stream."""

    return send_command(COMMAND_START_RECORD, record=record, **kwargs)


def stop_stream(**kwargs) -> dict:
    """Ask the agent to stop the running stream, if any."""

    return send_command(COMMAND_STOP, **kwargs)


def request_status(**kwargs) -> dict:
    """Ask the agent what it is currently doing."""

    return send_command(COMMAND_STATUS, **kwargs)


def _read_frame(connection: socket.socket) -> bytes:
    buffer = bytearray()

    while FRAME_DELIMITER not in buffer:
        data = connection.recv(1024)

        if not data:
            raise ControlClientError(
                "the Pi control agent closed the connection without replying"
            )

        buffer.extend(data)

        if len(buffer) > MAX_FRAME_BYTES:
            raise ControlClientError("the Pi control agent sent an oversized response")

    return bytes(buffer).split(FRAME_DELIMITER, 1)[0]
