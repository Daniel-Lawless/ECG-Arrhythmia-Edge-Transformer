import logging

from ecg_arrhythmia.streaming.onnx_sequence_classifier import (
    ONNXSequenceClassifier,
    PredictionEvent,
)
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine

logger = logging.getLogger(__name__)


class StreamingPredictor:
    """
    Turn a stream of SampleChunks into a stream of PredictionEvents.

    Composition rather than inheritance: StreamingEngine keeps validating,
    detecting and assembling, and this class only adds the model step.
    Each call returns just the predictions that call produced, and no
    prediction history is retained, so nothing can leak between records.
    """

    def __init__(
        self,
        engine: StreamingEngine,
        classifier: ONNXSequenceClassifier,
    ) -> None:
        self._engine = engine
        self._classifier = classifier

    @property
    def engine(self) -> StreamingEngine:
        """The engine being driven, for callers that need stream state."""

        return self._engine

    def start_record(
        self,
        record_name: str | None = None,
        start_index: int = 0,
    ) -> None:
        """Begin a new record, discarding all previous streaming state."""

        self._engine.start_record(
            record_name=record_name,
            start_index=start_index,
        )
        logger.info("Started inference for record %s", record_name)

    def reset(self) -> None:
        """Discard all streaming state without naming a new record."""

        self._engine.reset()

    def process_chunk(self, chunk: SampleChunk) -> list[PredictionEvent]:
        """
        Classify every sequence this chunk completed.

        A chunk may complete none, one or several sequences, and the
        predictions are returned in the order the engine emitted them.
        """

        # self._engine.process_chunk(chunk) passes the incoming ECG
        # chunk through the existing streaming pipeline. That may produce
        # [] if no sequence is ready yet, or [sequence1, sequence2, ...]
        # if one or more complete BeatSequences became available.
        # Then those sequences are immediately passed into _classify()
        return self._classify(self._engine.process_chunk(chunk))

    def flush(self) -> list[PredictionEvent]:
        """Classify the sequences released at the end of a finite record."""

        return self._classify(self._engine.flush())

    def _classify(self, sequences: list) -> list[PredictionEvent]:
        # Classify iterates through the sequences and returns a PredictionEvent
        # for each sequence.
        return [self._classifier.predict(sequence) for sequence in sequences]
