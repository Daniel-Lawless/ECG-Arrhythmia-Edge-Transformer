from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.deployment.verify_onnx_parity import load_pytorch_model
from ecg_arrhythmia.evaluation.evaluate_streaming_inference_parity import (
    ABSOLUTE_TOLERANCE,
    DROPOUT,
    NUM_LAYERS,
    RELATIVE_TOLERANCE,
    _event_logits,
    _pytorch_logits,
    _RecordingClassifier,
    compare_logits,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplaySource
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.streaming.streaming_predictor import StreamingPredictor

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = REPO_ROOT / "artifacts/models/ecg_sequence_transformer_tuned.pt"
ONNX_MODEL_PATH = REPO_ROOT / "artifacts/models/ecg_sequence_transformer.onnx"

RECORD_NAME = "114"

# Enough ECG to clear the ten-second warm-up and emit many sequences,
# without replaying a full thirty-minute record in a test.
REPLAY_SECONDS = 60.0

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_PATH.exists() and ONNX_MODEL_PATH.exists()),
    reason="Tuned PyTorch checkpoint and exported ONNX model required.",
)


@pytest.mark.integration
def test_all_three_inference_paths_agree_on_real_streaming_sequences():
    signals, fields, _ = load_record(record_name=RECORD_NAME)
    signal, _ = select_signal_channel(signals=signals, fields=fields)
    sampling_rate = float(fields["fs"])

    # Stream the opening minute of the record through the real pipeline.
    classifier = _RecordingClassifier(ONNX_MODEL_PATH)
    predictor = StreamingPredictor(
        engine=StreamingEngine(),
        classifier=classifier,
    )
    predictor.start_record(record_name=RECORD_NAME)

    source = ReplaySource(
        signal=signal[: int(REPLAY_SECONDS * sampling_rate)],
        sampling_rate=sampling_rate,
        chunk_size=DEFAULT_CHUNK_SIZE,
        record_name=RECORD_NAME,
    )

    for chunk in source.iter_chunks():
        predictor.process_chunk(chunk)

    predictor.flush()

    sequences = classifier.sequences
    assert len(sequences) >= 1

    # The other two paths consume the identical captured sequences.
    model = load_pytorch_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    offline_classifier = ONNXSequenceClassifier(ONNX_MODEL_PATH)
    offline_events = [offline_classifier.predict(sequence) for sequence in sequences]

    pytorch_logits = _pytorch_logits(model, sequences)
    offline_logits = _event_logits(offline_events)
    streaming_logits = _event_logits(classifier.events)
    target_peaks = np.asarray(
        [event.target_peak_index for event in classifier.events],
        dtype=np.int64,
    )

    for reference, comparison in (
        (pytorch_logits, offline_logits),
        (offline_logits, streaming_logits),
        (pytorch_logits, streaming_logits),
    ):
        result = compare_logits(
            reference=reference,
            comparison=comparison,
            target_peaks=target_peaks,
            relative_tolerance=RELATIVE_TOLERANCE,
            absolute_tolerance=ABSOLUTE_TOLERANCE,
        )

        assert result["class_agreement_percentage"] == 100.0
        assert result["num_arrays_outside_tolerance"] == 0
        assert result["passed"] is True


@pytest.mark.integration
def test_the_exported_model_matches_the_beat_contract():
    # Constructing the classifier validates the ONNX input and output
    # metadata against the 5-beat, 240-sample, four-class contract.
    classifier = ONNXSequenceClassifier(ONNX_MODEL_PATH)

    assert classifier.providers == ("CPUExecutionProvider",)
