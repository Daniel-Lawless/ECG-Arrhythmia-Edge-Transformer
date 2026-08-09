import numpy as np

from ecg_arrhythmia.data.label_mapping import INDEX_TO_LABEL, NUM_CLASSES
from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SEQUENCE_LENGTH,
    WINDOW_SIZE,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence
from ecg_arrhythmia.streaming.streaming_detector import DetectorTiming
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.streaming.streaming_predictor import StreamingPredictor

SAMPLING_RATE = 360.0

# Fast timings so a synthetic record exercises the whole pipeline.
FAST_TIMING = DetectorTiming(
    analysis_window_seconds=30.0,
    stride_seconds=1.0,
    warmup_seconds=1.0,
    confirmation_seconds=0.5,
)
SYNTHETIC_LENGTH = 6000
SYNTHETIC_PEAKS = [300, 660, 1020, 1380, 1740, 2100, 2460]


class CountingClassifier:
    """Classifier stand-in that records how often it was asked to predict."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, sequence: BeatSequence) -> PredictionEvent:
        self.calls += 1
        class_index = self.calls % NUM_CLASSES

        logits = np.zeros(NUM_CLASSES, dtype=np.float32)
        logits[class_index] = 1.0
        logits.setflags(write=False)

        return PredictionEvent(
            target_peak_index=int(sequence.target_peak_index),
            peak_indices=tuple(sequence.peak_indices),
            logits=logits,
            predicted_class_index=class_index,
            predicted_label=INDEX_TO_LABEL[class_index],
        )


class ScriptedEngine:
    """Engine stand-in that emits a scripted list of sequences per chunk."""

    def __init__(self, per_chunk, on_flush=()) -> None:
        self._per_chunk = [list(batch) for batch in per_chunk]
        self._on_flush = list(on_flush)
        self.started_records: list[tuple] = []
        self.resets = 0

    def start_record(self, record_name=None, start_index=0) -> None:
        self.started_records.append((record_name, start_index))

    def reset(self) -> None:
        self.resets += 1

    def process_chunk(self, chunk):
        return self._per_chunk.pop(0) if self._per_chunk else []

    def flush(self):
        return self._on_flush


def _sequence(target_peak_index: int) -> BeatSequence:
    return BeatSequence(
        ecg=np.zeros((SEQUENCE_LENGTH, 1, WINDOW_SIZE), dtype=np.float64),
        rr=np.ones((SEQUENCE_LENGTH, 2), dtype=np.float64),
        target_peak_index=target_peak_index,
        peak_indices=(1, 2, 3, 4, target_peak_index),
    )


def _chunk(start_index: int = 0, num_samples: int = 4) -> SampleChunk:
    return SampleChunk(
        samples=np.zeros(num_samples, dtype=np.float64),
        start_index=start_index,
        sampling_rate=SAMPLING_RATE,
    )


# ---------------------------------------------------------------------
#                        Sequence To Prediction
# ---------------------------------------------------------------------


def test_a_chunk_with_no_sequences_produces_no_predictions():
    predictor = StreamingPredictor(
        engine=ScriptedEngine(per_chunk=[[]]),
        classifier=CountingClassifier(),
    )

    assert predictor.process_chunk(_chunk()) == []


def test_a_chunk_with_one_sequence_produces_one_prediction():
    predictor = StreamingPredictor(
        engine=ScriptedEngine(per_chunk=[[_sequence(1940)]]),
        classifier=CountingClassifier(),
    )

    [event] = predictor.process_chunk(_chunk())

    assert event.target_peak_index == 1940


def test_several_sequences_from_one_chunk_are_classified_in_order():
    sequences = [_sequence(peak) for peak in (1940, 2300, 2660)]
    predictor = StreamingPredictor(
        engine=ScriptedEngine(per_chunk=[sequences]),
        classifier=CountingClassifier(),
    )

    events = predictor.process_chunk(_chunk())

    assert [event.target_peak_index for event in events] == [1940, 2300, 2660]


def test_sequences_released_by_flush_are_classified():
    predictor = StreamingPredictor(
        engine=ScriptedEngine(per_chunk=[[]], on_flush=[_sequence(5000)]),
        classifier=CountingClassifier(),
    )

    assert predictor.process_chunk(_chunk()) == []

    [event] = predictor.flush()

    assert event.target_peak_index == 5000


def test_each_call_returns_only_its_own_predictions():
    engine = ScriptedEngine(per_chunk=[[_sequence(100)], [_sequence(200)]])
    predictor = StreamingPredictor(engine=engine, classifier=CountingClassifier())

    first = predictor.process_chunk(_chunk(0))
    second = predictor.process_chunk(_chunk(4))

    assert [event.target_peak_index for event in first] == [100]
    assert [event.target_peak_index for event in second] == [200]


def test_record_lifecycle_calls_are_forwarded_to_the_engine():
    engine = ScriptedEngine(per_chunk=[])
    predictor = StreamingPredictor(engine=engine, classifier=CountingClassifier())

    predictor.start_record("114")
    predictor.reset()

    assert engine.started_records == [("114", 0)]
    assert engine.resets == 1


# ---------------------------------------------------------------------
#                     Against The Real Streaming Engine
# ---------------------------------------------------------------------


class _MarkerDetector(RPeakDetector):
    """Fake detector that treats every sample equal to 1.0 as an R-peak."""

    @property
    def name(self):
        return "marker"

    def _detect(self, signal, sampling_rate):
        return np.flatnonzero(signal == 1.0).astype(np.int64)


def _synthetic_record():
    signal = np.zeros(SYNTHETIC_LENGTH, dtype=np.float64)
    signal[SYNTHETIC_PEAKS] = 1.0
    return signal


def _replay(predictor: StreamingPredictor, signal, chunk_size: int = 36):
    events: list[PredictionEvent] = []

    for start_index in range(0, len(signal), chunk_size):
        events.extend(
            predictor.process_chunk(
                SampleChunk(
                    samples=signal[start_index : start_index + chunk_size],
                    start_index=start_index,
                    sampling_rate=SAMPLING_RATE,
                )
            )
        )

    events.extend(predictor.flush())

    return events


def _real_predictor(classifier: CountingClassifier) -> StreamingPredictor:
    return StreamingPredictor(
        engine=StreamingEngine(detector=_MarkerDetector(), timing=FAST_TIMING),
        classifier=classifier,
    )


def test_every_emitted_sequence_is_classified_exactly_once():
    classifier = CountingClassifier()
    predictor = _real_predictor(classifier)
    predictor.start_record("synthetic")

    events = _replay(predictor, _synthetic_record())

    assert [event.target_peak_index for event in events] == SYNTHETIC_PEAKS[-2:]
    # One prediction per emitted sequence, from the one classifier the
    # predictor was built with: nothing is rebuilt per chunk or sequence.
    assert classifier.calls == len(events)


def test_starting_a_new_record_leaves_no_prediction_state_behind():
    classifier = CountingClassifier()
    predictor = _real_predictor(classifier)
    signal = _synthetic_record()

    predictor.start_record("synthetic")
    first = _replay(predictor, signal)

    predictor.start_record("synthetic-again")
    second = _replay(predictor, signal)

    assert [event.target_peak_index for event in second] == [
        event.target_peak_index for event in first
    ]
    assert [event.peak_indices for event in second] == [
        event.peak_indices for event in first
    ]
