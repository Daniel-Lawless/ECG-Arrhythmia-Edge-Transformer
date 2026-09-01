import json

from ecg_arrhythmia.data.mitdb_records import MITDB_RECORDS, PACED_RECORDS

ENCODING = "utf-8"
FRAME_DELIMITER = b"\n"

# The demo-record allowlist, derived from the project's existing record
# constants rather than duplicating a list. Paced records are excluded
# because paced beats were never part of the four-class AAMI target the
# deployed model was trained on, so replaying one would demonstrate
# confident predictions the model has no basis to make.
DEMO_RECORDS = tuple(record for record in MITDB_RECORDS if record not in PACED_RECORDS)

# The record used when the dashboard has no other preference: the
# high-activity stress record from Section 6.4 validation.
DEFAULT_DEMO_RECORD = "233"

COMMAND_START_RECORD = "start_record"
COMMAND_STOP = "stop"
COMMAND_STATUS = "status"

SUPPORTED_COMMANDS = (COMMAND_START_RECORD, COMMAND_STOP, COMMAND_STATUS)

# Commands that carry a record name; every other command must not.
COMMANDS_REQUIRING_RECORD = (COMMAND_START_RECORD,)

STATUS_OK = "ok"
STATUS_ERROR = "error"

# A control frame is a handful of short fields; anything larger is not
# this protocol and is refused rather than buffered.
MAX_FRAME_BYTES = 4096


class ControlProtocolError(ValueError):
    """A control frame is malformed, unsupported or disallowed."""


def encode_request(command: str, record: str | None = None) -> bytes:
    """
    Build one control request frame.

    Validates before encoding so a client cannot put an unsupported
    command or an unknown record onto the wire in the first place.
    """

    if command not in SUPPORTED_COMMANDS:
        supported = ", ".join(SUPPORTED_COMMANDS)
        raise ControlProtocolError(
            f"unsupported command {command!r}; supported commands are {supported}"
        )

    payload: dict[str, str] = {"command": command}

    if command in COMMANDS_REQUIRING_RECORD:
        if record is None:
            raise ControlProtocolError(f"command {command!r} requires a record")

        payload["record"] = validated_record(record)
    elif record is not None:
        raise ControlProtocolError(f"command {command!r} does not take a record")

    return _encode(payload)


def encode_response(
    status: str,
    message: str,
    running: bool,
    record: str | None = None,
) -> bytes:
    """
    Build one control response frame.

    `running` and `record` describe the agent's state after handling
    the request, so a client never has to infer it from prose.
    """

    if status not in (STATUS_OK, STATUS_ERROR):
        raise ControlProtocolError(f"unsupported response status {status!r}")

    return _encode(
        {
            "status": status,
            "message": str(message),
            "running": bool(running),
            "record": None if record is None else str(record),
        }
    )


def decode_request(frame: bytes | str) -> dict:
    """
    Decode and fully validate one control request.

    Every field is checked here, on the Pi, even though the dashboard
    offers a dropdown: the dropdown is a usability affordance, not a
    security boundary. Returns a dict with 'command' and, for commands
    that take one, 'record'.
    """

    message = _decode_object(frame)
    command = message.get("command")

    if not isinstance(command, str):
        raise ControlProtocolError(
            f"'command' must be a string, found {type(command).__name__}"
        )

    if command not in SUPPORTED_COMMANDS:
        supported = ", ".join(SUPPORTED_COMMANDS)
        raise ControlProtocolError(
            f"unsupported command {command!r}; supported commands are {supported}"
        )

    allowed = {"command"}

    if command in COMMANDS_REQUIRING_RECORD:
        allowed.add("record")

    # Strict contract: unexpected fields are rejected, not ignored.
    # Silently dropping them would make a request that tried to set a
    # destination or model path look like it had been accepted as
    # written, and would let the wire format drift without anyone
    # noticing. A field this protocol does not define is an error.
    unexpected = sorted(set(message) - allowed)

    if unexpected:
        raise ControlProtocolError(
            f"command {command!r} does not accept fields: " + ", ".join(unexpected)
        )

    request = {"command": command}

    if command in COMMANDS_REQUIRING_RECORD:
        if "record" not in message:
            raise ControlProtocolError(f"command {command!r} requires a record")

        request["record"] = validated_record(message["record"])

    return request


def decode_response(frame: bytes | str) -> dict:
    """Decode and validate one control response."""

    message = _decode_object(frame)

    for field in ("status", "message", "running"):
        if field not in message:
            raise ControlProtocolError(f"response is missing required field {field!r}")

    if message["status"] not in (STATUS_OK, STATUS_ERROR):
        raise ControlProtocolError(f"unsupported response status {message['status']!r}")

    if not isinstance(message["running"], bool):
        raise ControlProtocolError("response field 'running' must be a boolean")

    return message


def validated_record(record) -> str:
    """
    Return the record name if it is a known demo record, else raise.

    Membership of a fixed enum is the whole validation: no normalising,
    no path handling, no pattern matching that could be coaxed into
    accepting something adjacent to a real record name.
    """

    if not isinstance(record, str):
        raise ControlProtocolError(
            f"'record' must be a string, found {type(record).__name__}"
        )

    if record not in DEMO_RECORDS:
        raise ControlProtocolError(f"record {record!r} is not an available demo record")

    return record


def _encode(payload: dict) -> bytes:
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

    return text.encode(ENCODING) + FRAME_DELIMITER


def _decode_object(frame: bytes | str) -> dict:
    if isinstance(frame, bytes):
        try:
            frame = frame.decode(ENCODING)
        except UnicodeDecodeError as error:
            raise ControlProtocolError(
                f"control frame is not valid UTF-8: {error}"
            ) from error

    try:
        message = json.loads(frame)
    except json.JSONDecodeError as error:
        raise ControlProtocolError(
            f"control frame is not valid JSON: {error}"
        ) from error

    if not isinstance(message, dict):
        raise ControlProtocolError(
            f"control frame must decode to a JSON object, "
            f"found {type(message).__name__}"
        )

    return message
