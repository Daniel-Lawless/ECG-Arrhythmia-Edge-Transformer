import socket

import numpy as np
import pytest

from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.transport.protocol import (
    ProtocolError,
    TransportError,
    encode_prediction,
    encode_sample_chunk,
)
from ecg_arrhythmia.transport.tcp_receiver import FrameAssembler, TCPStreamReceiver
from ecg_arrhythmia.transport.tcp_sender import TCPStreamSender

# Generous test timeout: only ever hit on failure, never on success,
# so it cannot make the suite flaky.
TEST_TIMEOUT_SECONDS = 5.0


def _chunk(start_index: int = 0) -> SampleChunk:
    return SampleChunk(
        samples=np.full(36, 0.125, dtype=np.float64),
        start_index=start_index,
        sampling_rate=360.0,
    )


def _event(target: int = 2358) -> PredictionEvent:
    return PredictionEvent(
        target_peak_index=target,
        peak_indices=(2000, target, 2799),
        logits=np.array([3.5, -1.25, 0.5, -2.0], dtype=np.float32),
        predicted_class_index=0,
        predicted_label="N",
    )


# ---------------------------------------------------------------------
#                          Frame Assembly
# ---------------------------------------------------------------------


def test_two_frames_arriving_in_one_receive_are_both_released():
    first = encode_sample_chunk(_chunk(0), "114")
    second = encode_sample_chunk(_chunk(36), "114")

    frames = FrameAssembler().feed(first + second)

    assert frames == [first.rstrip(b"\n"), second.rstrip(b"\n")]


def test_one_frame_split_across_many_receives_is_reassembled():
    frame = encode_prediction(_event(), "114")
    assembler = FrameAssembler()
    released = []

    # Worst-case fragmentation: one byte per recv().
    for offset in range(len(frame)):
        released.extend(assembler.feed(frame[offset : offset + 1]))

    assert released == [frame.rstrip(b"\n")]
    assert assembler.pending_bytes == 0


def test_a_partial_frame_stays_buffered_until_its_newline_arrives():
    frame = encode_sample_chunk(_chunk(0), "114")
    assembler = FrameAssembler()

    assert assembler.feed(frame[:10]) == []
    assert assembler.pending_bytes == 10
    assert assembler.feed(frame[10:]) == [frame.rstrip(b"\n")]
    assert assembler.pending_bytes == 0


def test_a_mixture_of_complete_and_partial_frames_is_handled():
    first = encode_sample_chunk(_chunk(0), "114")
    second = encode_prediction(_event(), "114")
    assembler = FrameAssembler()

    assert assembler.feed(first + second[:7]) == [first.rstrip(b"\n")]
    assert assembler.feed(second[7:]) == [second.rstrip(b"\n")]


def test_an_endless_frame_without_a_newline_fails_clearly():
    assembler = FrameAssembler(max_frame_bytes=64)

    with pytest.raises(ProtocolError, match="without a\\s+newline"):
        assembler.feed(b"x" * 65)


# ---------------------------------------------------------------------
#                        Sender Error Handling
# ---------------------------------------------------------------------


def test_sending_before_connecting_raises_a_transport_error():
    sender = TCPStreamSender(host="127.0.0.1", port=1)

    with pytest.raises(TransportError, match="not connected"):
        sender.send_sample_chunk(_chunk(), record_name="114")


def test_connecting_to_a_closed_port_raises_a_transport_error():
    # Bind an ephemeral port, then close it so nothing is listening.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()

    sender = TCPStreamSender(
        host="127.0.0.1",
        port=closed_port,
        connect_timeout=1.0,
    )

    with pytest.raises(TransportError, match="could not connect"):
        sender.connect()


def test_closing_twice_is_safe_and_connected_reflects_lifecycle():
    sender = TCPStreamSender(host="127.0.0.1", port=1)

    assert not sender.connected

    sender.close()
    sender.close()

    assert not sender.connected


# ---------------------------------------------------------------------
#                    Localhost Sender <-> Receiver
# ---------------------------------------------------------------------


def test_messages_over_one_persistent_connection_arrive_in_order():
    with TCPStreamReceiver(
        host="127.0.0.1",
        port=0,
        timeout=TEST_TIMEOUT_SECONDS,
    ) as receiver:
        # The listener's backlog holds the connection until messages()
        # accepts it, so no thread is needed and nothing is timing
        # dependent.
        with TCPStreamSender(host="127.0.0.1", port=receiver.bound_port) as sender:
            sender.send_sample_chunk(_chunk(0), record_name="114")
            sender.send_prediction(_event(370), record_name="114")
            sender.send_sample_chunk(_chunk(36), record_name="114")

        received = list(receiver.messages())

    assert [message["message_type"] for message in received] == [
        "sample_chunk",
        "prediction",
        "sample_chunk",
    ]
    assert received[0]["start_index"] == 0
    assert received[1]["target_peak_index"] == 370
    assert received[2]["start_index"] == 36
    assert all(message["record_name"] == "114" for message in received)


def test_a_clean_disconnect_ends_the_message_stream_without_error():
    with TCPStreamReceiver(
        host="127.0.0.1",
        port=0,
        timeout=TEST_TIMEOUT_SECONDS,
    ) as receiver:
        with TCPStreamSender(host="127.0.0.1", port=receiver.bound_port) as sender:
            sender.send_sample_chunk(_chunk(0), record_name="114")

        received = list(receiver.messages())

    assert len(received) == 1


def test_a_disconnect_mid_frame_raises_a_protocol_error():
    with TCPStreamReceiver(
        host="127.0.0.1",
        port=0,
        timeout=TEST_TIMEOUT_SECONDS,
    ) as receiver:
        raw = socket.create_connection(("127.0.0.1", receiver.bound_port))
        raw.sendall(b'{"schema_version":1,"message_type":"sample_chunk"')
        raw.close()

        with pytest.raises(ProtocolError, match="closed mid-frame"):
            list(receiver.messages())


def test_an_invalid_frame_from_the_peer_raises_a_protocol_error():
    with TCPStreamReceiver(
        host="127.0.0.1",
        port=0,
        timeout=TEST_TIMEOUT_SECONDS,
    ) as receiver:
        raw = socket.create_connection(("127.0.0.1", receiver.bound_port))
        raw.sendall(b'{"schema_version":999,"message_type":"sample_chunk"}\n')
        raw.close()

        with pytest.raises(ProtocolError, match="unsupported schema version"):
            list(receiver.messages())


def test_receiving_before_listening_raises_a_transport_error():
    receiver = TCPStreamReceiver(host="127.0.0.1", port=0)

    with pytest.raises(TransportError, match="not listening"):
        next(receiver.messages())
