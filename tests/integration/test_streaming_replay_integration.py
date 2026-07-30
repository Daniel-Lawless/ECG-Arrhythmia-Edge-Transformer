import numpy as np

from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.evaluation.replay_streaming_record import replay_record
from ecg_arrhythmia.streaming.replay_source import ReplayMode, ReplaySource
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine

# 114 is one of the existing XQRS-centred validation records.
VALIDATION_RECORD = "114"
CHUNK_SIZE = 360


def test_accelerated_replay_of_a_real_validation_record():
    source = ReplaySource.from_record(
        record_name=VALIDATION_RECORD,
        chunk_size=CHUNK_SIZE,
        mode=ReplayMode.ACCELERATED,
    )

    summary = replay_record(source=source, engine=StreamingEngine())

    assert summary.record_name == VALIDATION_RECORD
    assert summary.replay_mode == ReplayMode.ACCELERATED.value
    assert summary.sampling_rate == 360.0
    assert summary.chunk_size == CHUNK_SIZE

    # Every sample of the record reaches the engine exactly once.
    assert summary.total_samples_accepted == summary.total_input_samples
    assert summary.total_emitted_chunks == source.num_chunks
    assert summary.first_sample_index == 0
    assert summary.final_sample_index == summary.total_input_samples - 1
    assert summary.continuity_validated is True
    assert summary.elapsed_seconds >= 0.0


def test_replayed_samples_match_the_offline_signal():
    signals, fields, _ = load_record(record_name=VALIDATION_RECORD)
    expected_signal, expected_lead = select_signal_channel(
        signals=signals,
        fields=fields,
    )

    source = ReplaySource.from_record(
        record_name=VALIDATION_RECORD,
        chunk_size=CHUNK_SIZE,
    )

    # The streaming path must deliver the identical lead and samples the
    # offline pipeline uses, in the same order.
    assert source.lead_name == expected_lead

    reassembled = np.concatenate([chunk.samples for chunk in source.iter_chunks()])
    np.testing.assert_array_equal(reassembled, expected_signal)
