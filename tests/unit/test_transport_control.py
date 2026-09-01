import json
import threading

import pytest

from ecg_arrhythmia.dashboard.record_control import (
    default_record_index,
    describe_response,
)
from ecg_arrhythmia.transport.control_client import (
    DEFAULT_READ_TIMEOUT_SECONDS,
    ControlClientError,
    request_status,
    start_record,
    stop_stream,
)
from ecg_arrhythmia.transport.control_protocol import (
    COMMAND_START_RECORD,
    COMMAND_STATUS,
    COMMAND_STOP,
    DEFAULT_DEMO_RECORD,
    DEMO_RECORDS,
    STATUS_ERROR,
    STATUS_OK,
    ControlProtocolError,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from ecg_arrhythmia.transport.control_server import (
    STREAM_STOP_TIMEOUT_SECONDS,
    ControlServer,
    RecordStreamRunner,
)

# ---------------------------------------------------------------------
#                          Record Allowlist
# ---------------------------------------------------------------------


# Modules that must remain importable on the dashboard machine, which
# has no WFDB: replay and inference happen on the Pi and the PC only
# needs record names and a socket.
DASHBOARD_SIDE_MODULES = [
    "ecg_arrhythmia.transport.control_config",
    "ecg_arrhythmia.transport.control_protocol",
    "ecg_arrhythmia.transport.control_client",
    "ecg_arrhythmia.dashboard.record_control",
]

# Importing any of the above must not reach the Pi-side stack. The
# control server is listed because it is the doorway to the rest:
# importing it pulls replay_source -> load_record -> wfdb.
PI_SIDE_MODULES = [
    "wfdb",
    "ecg_arrhythmia.transport.control_server",
    "ecg_arrhythmia.streaming.replay_source",
    "ecg_arrhythmia.data.build_dataset",
]


@pytest.mark.parametrize("module", DASHBOARD_SIDE_MODULES)
def test_dashboard_side_modules_do_not_load_the_pi_dependency_stack(module):
    # A clean subprocess is the only honest check: within pytest the
    # Pi-side modules are already imported by other tests, so
    # sys.modules here would never reveal the leak.
    import os
    import subprocess
    import sys
    from pathlib import Path

    # pytest's pythonpath setting does not reach a subprocess, so src
    # is passed explicitly rather than relying on an editable install.
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    forbidden = ", ".join(repr(name) for name in PI_SIDE_MODULES)
    program = (
        f"import sys;"
        f"import {module};"
        f"loaded = set(sys.modules);"
        f"leaked = [name for name in ({forbidden},) if name in loaded];"
        f"assert not leaked, '{module} pulled in ' + ', '.join(leaked);"
        f"print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_both_control_sides_share_one_port_constant():
    # The port must have exactly one definition: a client and server
    # that disagree would fail only at runtime, on hardware.
    from ecg_arrhythmia.transport import (
        control_client,
        control_config,
        control_server,
    )

    assert control_config.DEFAULT_CONTROL_PORT == 8767
    assert control_client.DEFAULT_CONTROL_PORT is control_config.DEFAULT_CONTROL_PORT
    assert control_server.DEFAULT_CONTROL_PORT is control_config.DEFAULT_CONTROL_PORT


def test_the_record_constants_have_a_single_source_of_truth():
    # build_dataset re-exports rather than redefining, so the two
    # consumers can never drift apart.
    from ecg_arrhythmia.data import build_dataset, mitdb_records

    assert build_dataset.MITDB_RECORDS is mitdb_records.MITDB_RECORDS
    assert build_dataset.PACED_RECORDS is mitdb_records.PACED_RECORDS


def test_the_demo_records_exclude_paced_records():
    # Paced beats fall outside the four AAMI classes the deployed model
    # was trained on, so they are deliberately not offered.
    for paced in ("102", "104", "107", "217"):
        assert paced not in DEMO_RECORDS

    assert len(DEMO_RECORDS) == 44
    assert DEFAULT_DEMO_RECORD in DEMO_RECORDS
    # The records used throughout Sections 5 and 6 remain available.
    for record in ("114", "122", "209", "210", "231", "233"):
        assert record in DEMO_RECORDS


# ---------------------------------------------------------------------
#                         Control Protocol
# ---------------------------------------------------------------------


def test_a_start_request_round_trips():
    request = decode_request(encode_request(COMMAND_START_RECORD, "233"))

    assert request == {"command": COMMAND_START_RECORD, "record": "233"}


@pytest.mark.parametrize("command", [COMMAND_STOP, COMMAND_STATUS])
def test_recordless_commands_round_trip(command):
    assert decode_request(encode_request(command)) == {"command": command}


def test_a_response_round_trips_with_agent_state():
    response = decode_response(
        encode_response(STATUS_OK, "started record 233", running=True, record="233")
    )

    assert response["status"] == STATUS_OK
    assert response["message"] == "started record 233"
    assert response["running"] is True
    assert response["record"] == "233"


@pytest.mark.parametrize(
    "record",
    [
        "999",
        "102",  # a real MIT-BIH record, but paced and so not offered
        "",
        "../../etc/passwd",
        "233; rm -rf /",
        "233\n",
        "__import__('os').system('id')",
    ],
)
def test_records_outside_the_allowlist_are_rejected(record):
    # Neither side will put an unknown record on the wire, and the Pi
    # rejects one regardless of how it was produced.
    with pytest.raises(ControlProtocolError):
        encode_request(COMMAND_START_RECORD, record)

    frame = json.dumps({"command": COMMAND_START_RECORD, "record": record})

    with pytest.raises(ControlProtocolError, match="not an available demo record"):
        decode_request(frame)


@pytest.mark.parametrize("record", [233, None, True, ["233"], {"record": "233"}])
def test_non_string_records_are_rejected(record):
    frame = json.dumps({"command": COMMAND_START_RECORD, "record": record})

    with pytest.raises(ControlProtocolError):
        decode_request(frame)


def test_unsupported_commands_are_rejected():
    frame = json.dumps({"command": "run_shell", "record": "233"})

    with pytest.raises(ControlProtocolError, match="unsupported command"):
        decode_request(frame)


def test_a_start_request_without_a_record_is_rejected():
    with pytest.raises(ControlProtocolError, match="requires a record"):
        decode_request(json.dumps({"command": COMMAND_START_RECORD}))


def test_malformed_control_frames_are_rejected():
    with pytest.raises(ControlProtocolError, match="not valid JSON"):
        decode_request(b'{"command": ')

    with pytest.raises(ControlProtocolError, match="JSON object"):
        decode_request(b"[1, 2, 3]")

    with pytest.raises(ControlProtocolError, match="not valid UTF-8"):
        decode_request(b"\xff\xfe{}")


def test_unexpected_request_fields_are_rejected_not_ignored():
    # Strict contract: a request trying to set a destination or model
    # path fails outright rather than succeeding with the extra fields
    # quietly dropped, which would look like they had been honoured.
    frame = json.dumps(
        {
            "command": COMMAND_START_RECORD,
            "record": "233",
            "host": "10.0.0.9",
            "port": 9999,
            "model_path": "/tmp/evil.onnx",
        }
    )

    with pytest.raises(ControlProtocolError, match="does not accept fields"):
        decode_request(frame)


@pytest.mark.parametrize("command", [COMMAND_STOP, COMMAND_STATUS])
def test_recordless_commands_reject_a_record_field(command):
    frame = json.dumps({"command": command, "record": "233"})

    with pytest.raises(ControlProtocolError, match="does not accept fields: record"):
        decode_request(frame)


def test_the_rejection_names_every_unexpected_field():
    frame = json.dumps(
        {"command": COMMAND_STOP, "model_path": "/tmp/x", "host": "10.0.0.9"}
    )

    with pytest.raises(ControlProtocolError) as error:
        decode_request(frame)

    assert "host" in str(error.value)
    assert "model_path" in str(error.value)


# ---------------------------------------------------------------------
#                       Stream Runner Lifecycle
# ---------------------------------------------------------------------


class FakeStream:
    """
    Stands in for run_record_stream: blocks until told to stop.

    Records the keyword arguments it was called with so the seam
    between the agent and the real streaming entry point is pinned.
    """

    def __init__(self) -> None:
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()

        should_stop = kwargs["should_stop"]

        # Mimic the real loop: poll the stop flag, finish when set.
        while not should_stop():
            if self.release.wait(timeout=0.01):
                break

        return {"chunks_sent": 1, "stopped_early": should_stop()}


def _runner(stream=None) -> RecordStreamRunner:
    return RecordStreamRunner(
        host="192.0.2.1",
        port=8765,
        stream_callable=stream or FakeStream(),
    )


def test_starting_a_record_runs_the_stream_on_a_worker_thread():
    stream = FakeStream()
    runner = _runner(stream)

    try:
        message = runner.start("233")

        assert stream.started.wait(timeout=2.0)
        # start() returned while the stream is still running: the
        # control connection is never held for the record's duration.
        assert runner.running is True
        assert runner.current_record == "233"
        # Acceptance wording only: the worker exists, but the model
        # session, record fetch and data connection may not.
        assert message == "start accepted for record 233"
        assert "started" not in message
    finally:
        runner.stop()


def test_the_runner_passes_the_fixed_destination_not_the_request():
    stream = FakeStream()
    runner = _runner(stream)

    try:
        runner.start("114")

        assert stream.started.wait(timeout=2.0)
        (call,) = stream.calls

        # The destination comes from how the agent was started; the
        # control request only chose the record.
        assert call["host"] == "192.0.2.1"
        assert call["port"] == 8765
        assert call["record"] == "114"
        assert callable(call["should_stop"])
    finally:
        runner.stop()


def test_stopping_ends_the_stream_and_reports_the_record():
    runner = _runner()
    runner.start("233")

    assert runner.stop() == "233"
    assert runner.running is False
    assert runner.current_record is None


def test_stopping_an_idle_runner_reports_nothing_stopped():
    assert _runner().stop() is None


def test_starting_a_second_record_replaces_the_first():
    stream = FakeStream()
    runner = _runner(stream)

    try:
        runner.start("114")
        assert stream.started.wait(timeout=2.0)

        message = runner.start("233")

        # Exactly one stream runs at a time: a second concurrent sender
        # would stall against the single-client PC receiver.
        assert message == "stopped record 114; start accepted for record 233"
        assert runner.current_record == "233"
        assert [call["record"] for call in stream.calls] == ["114", "233"]
    finally:
        runner.stop()


def test_the_outgoing_worker_is_fully_joined_before_the_next_one_starts():
    # The PC receiver accepts one sender, so two overlapping workers
    # would leave the second stalled in the TCP backlog. Record when
    # each worker enters and leaves so overlap is directly observable.
    events = []
    lock = threading.Lock()

    def tracking_stream(**kwargs):
        record = kwargs["record"]
        should_stop = kwargs["should_stop"]

        with lock:
            events.append(("enter", record))

        while not should_stop():
            threading.Event().wait(0.01)

        with lock:
            events.append(("exit", record))

        return {"chunks_sent": 1}

    runner = RecordStreamRunner(host="192.0.2.1", stream_callable=tracking_stream)

    try:
        runner.start("114")

        for _ in range(200):
            if events:
                break

            threading.Event().wait(0.01)

        runner.start("233")

        for _ in range(200):
            if len(events) >= 3:
                break

            threading.Event().wait(0.01)

        # The first worker must have exited before the second entered.
        assert events[:3] == [("enter", "114"), ("exit", "114"), ("enter", "233")]
    finally:
        runner.stop()


def test_a_failing_stream_is_recorded_rather_than_lost():
    def failing_stream(**kwargs):
        raise OSError("connection refused")

    runner = RecordStreamRunner(host="192.0.2.1", stream_callable=failing_stream)
    runner.start("233")
    runner.stop()

    assert "connection refused" in runner.last_error
    assert "connection refused" in runner.status_message()
    assert runner.running is False


# ---------------------------------------------------------------------
#                    Server and Client over Loopback
# ---------------------------------------------------------------------


def _serve_one(server: ControlServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_once, daemon=True)
    thread.start()

    return thread


def test_the_client_starts_a_record_through_a_real_socket():
    stream = FakeStream()
    runner = _runner(stream)
    server = ControlServer(runner, host="127.0.0.1", port=0)
    server.listen()

    try:
        thread = _serve_one(server)
        response = start_record("233", host="127.0.0.1", port=server.bound_port)
        thread.join(timeout=2.0)

        assert response["status"] == STATUS_OK
        assert response["running"] is True
        assert response["record"] == "233"
        assert stream.calls[0]["record"] == "233"
    finally:
        server.close()
        runner.stop()


def test_the_server_rejects_an_unknown_record_without_starting_anything():
    stream = FakeStream()
    runner = _runner(stream)
    server = ControlServer(runner, host="127.0.0.1", port=0)
    server.listen()

    try:
        thread = _serve_one(server)
        # Bypass the client's own validation to prove the Pi validates
        # independently of whatever the dashboard offers.
        import socket

        with socket.create_connection(("127.0.0.1", server.bound_port)) as connection:
            connection.sendall(
                json.dumps({"command": "start_record", "record": "999"}).encode()
                + b"\n"
            )
            frame = connection.recv(4096)

        thread.join(timeout=2.0)
        response = decode_response(frame.split(b"\n", 1)[0])

        assert response["status"] == STATUS_ERROR
        assert "not an available demo record" in response["message"]
        assert response["running"] is False
        assert stream.calls == []
    finally:
        server.close()
        runner.stop()


def test_stop_and_status_commands_work_over_the_socket():
    runner = _runner()
    server = ControlServer(runner, host="127.0.0.1", port=0)
    server.listen()
    port = server.bound_port

    try:
        thread = _serve_one(server)
        start_record("114", host="127.0.0.1", port=port)
        thread.join(timeout=2.0)

        thread = _serve_one(server)
        status = request_status(host="127.0.0.1", port=port)
        thread.join(timeout=2.0)

        assert status["running"] is True
        assert "114" in status["message"]

        thread = _serve_one(server)
        stopped = stop_stream(host="127.0.0.1", port=port)
        thread.join(timeout=2.0)

        assert stopped["status"] == STATUS_OK
        assert stopped["running"] is False
        assert stopped["message"] == "stopped record 114"
    finally:
        server.close()
        runner.stop()


def test_a_request_split_across_several_packets_is_reassembled():
    # TCP is a byte stream: one sendall on the client is not one recv
    # on the server. The frame must be reassembled from the newline,
    # never assumed to arrive whole.
    import socket
    import time

    stream = FakeStream()
    runner = _runner(stream)
    server = ControlServer(runner, host="127.0.0.1", port=0)
    server.listen()

    try:
        thread = _serve_one(server)
        request = json.dumps({"command": "start_record", "record": "233"}).encode()

        with socket.create_connection(("127.0.0.1", server.bound_port)) as connection:
            # One byte at a time, with the newline arriving last.
            for index in range(len(request)):
                connection.sendall(request[index : index + 1])
                time.sleep(0.001)

            connection.sendall(b"\n")
            frame = connection.recv(4096)

        thread.join(timeout=2.0)
        response = decode_response(frame.split(b"\n", 1)[0])

        assert response["status"] == STATUS_OK
        assert stream.calls[0]["record"] == "233"
    finally:
        server.close()
        runner.stop()


def test_a_request_closed_before_its_newline_starts_nothing():
    # An incomplete frame must never be parsed as if it were whole.
    import socket

    stream = FakeStream()
    runner = _runner(stream)
    server = ControlServer(runner, host="127.0.0.1", port=0)
    server.listen()

    try:
        thread = _serve_one(server)

        with socket.create_connection(("127.0.0.1", server.bound_port)) as connection:
            connection.sendall(b'{"command": "start_record", "record": "233"')

        thread.join(timeout=2.0)

        assert stream.calls == []
        assert runner.running is False
    finally:
        server.close()
        runner.stop()


def test_an_oversized_request_is_refused_without_unbounded_buffering():
    import socket

    stream = FakeStream()
    runner = _runner(stream)
    server = ControlServer(runner, host="127.0.0.1", port=0)
    server.listen()

    try:
        thread = _serve_one(server)

        with socket.create_connection(("127.0.0.1", server.bound_port)) as connection:
            connection.settimeout(5.0)
            # Far past MAX_FRAME_BYTES with no newline anywhere.
            try:
                connection.sendall(b"x" * 200_000)
            except OSError:
                # The server may refuse and close mid-write.
                pass

            frame = connection.recv(4096)

        thread.join(timeout=3.0)

        if frame:
            response = decode_response(frame.split(b"\n", 1)[0])

            assert response["status"] == STATUS_ERROR
            assert "too large" in response["message"]

        assert stream.calls == []
    finally:
        server.close()
        runner.stop()


def test_an_unreachable_agent_raises_a_reportable_client_error():
    # Port 1 on loopback: nothing listens, so connect fails fast.
    with pytest.raises(ControlClientError, match="could not reach the Pi"):
        start_record("233", host="127.0.0.1", port=1, connect_timeout=1.0)


def test_the_client_refuses_an_invalid_record_before_opening_a_socket():
    # Unroutable address: reaching the socket at all would hang, so a
    # prompt ControlProtocolError proves validation came first.
    with pytest.raises(ControlProtocolError):
        start_record("999", host="192.0.2.1", port=9, connect_timeout=0.1)


def test_the_read_timeout_outlasts_the_agents_own_stop_limit():
    # A start that replaces a running stream blocks on the agent's
    # join; the client must not give up before the agent itself would.
    assert DEFAULT_READ_TIMEOUT_SECONDS > STREAM_STOP_TIMEOUT_SECONDS


# ---------------------------------------------------------------------
#                     Dashboard-Facing Helpers
# ---------------------------------------------------------------------


def test_the_default_selection_is_the_default_demo_record():
    assert DEMO_RECORDS[default_record_index()] == DEFAULT_DEMO_RECORD


def test_an_absent_default_falls_back_to_the_first_record():
    assert default_record_index(("114", "233"), "999") == 0


def test_responses_are_described_without_rewriting_the_agent_message():
    severity, text = describe_response(
        {"status": STATUS_OK, "message": "started record 233"}
    )

    assert severity == "success"
    assert "started record 233" in text

    severity, text = describe_response(
        {"status": STATUS_ERROR, "message": "record '999' is not available"}
    )

    assert severity == "error"
    assert "999" in text
