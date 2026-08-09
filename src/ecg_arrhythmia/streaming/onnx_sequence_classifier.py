import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ecg_arrhythmia.data.label_mapping import INDEX_TO_LABEL, NUM_CLASSES
from ecg_arrhythmia.preprocessing.beat_extraction import (
    RR_FEATURE_DIM,
    SEQUENCE_LENGTH,
    WINDOW_SIZE,
)
from ecg_arrhythmia.streaming.onnx_contract import (
    ONNX_ECG_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    ONNX_RR_INPUT_NAME,
    create_onnx_session,
    validate_onnx_model,
    validate_session_contract,
)
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionEvent:
    """
    One classified beat sequence.

    target_peak_index and peak_indices are carried through unchanged from
    the BeatSequence, so a prediction can always be traced back to the
    detections that produced it. logits is the raw (4,) model output and
    is read-only: it is the parity contract for this phase, so nothing
    downstream may quietly rescale or overwrite it.
    """

    target_peak_index: int
    peak_indices: tuple[int, ...]
    logits: NDArray[np.float32]
    predicted_class_index: int
    predicted_label: str


class ONNXSequenceClassifier:
    """
    Classify streaming beat sequences with ONNX Runtime.

    The InferenceSession is created once during construction and reused
    for every sequence, because building it per prediction would dominate
    the cost and defeat the point of streaming inference.
    """

    def __init__(self, onnx_path: Path) -> None:
        validate_onnx_model(onnx_path)

        # onnx_contract owns the provider choice and every check of what
        # the deployed graph looks like.
        self._session: ort.InferenceSession = create_onnx_session(onnx_path)
        validate_session_contract(self._session)

        logger.info(
            "Loaded ONNX classifier from %s using providers %s",
            onnx_path,
            self.providers,
        )

    @property
    def providers(self) -> tuple[str, ...]:
        """Execution providers the session was created with."""

        return tuple(self._session.get_providers())

    def predict(self, sequence: BeatSequence) -> PredictionEvent:
        """Run one beat sequence through the model."""

        # Checks its shape, converts it to contiguous float32,
        # and adds a batch dimension.
        ecg = _as_model_input(
            sequence.ecg,
            expected_shape=(SEQUENCE_LENGTH, 1, WINDOW_SIZE),
            name="ECG",
        )
        rr = _as_model_input(
            sequence.rr,
            expected_shape=(SEQUENCE_LENGTH, RR_FEATURE_DIM),
            name="RR",
        )

        # Pass these inputs into our ONNX session
        outputs = self._session.run(
            output_names=[ONNX_OUTPUT_NAME],
            input_feed={
                ONNX_ECG_INPUT_NAME: ecg,
                ONNX_RR_INPUT_NAME: rr,
            },
        )
        # Validated logits.
        logits = _validate_logits(outputs[0])

        # Returns the index of largest logit
        predicted_class_index = int(np.argmax(logits))

        logger.debug(
            "Classified sequence with target peak %d as %s",
            sequence.target_peak_index,
            INDEX_TO_LABEL[predicted_class_index],
        )

        # One prediction represents a PredictionEvent
        return PredictionEvent(
            target_peak_index=int(sequence.target_peak_index),
            peak_indices=tuple(int(peak) for peak in sequence.peak_indices),
            logits=logits,
            predicted_class_index=predicted_class_index,
            predicted_label=INDEX_TO_LABEL[predicted_class_index],
        )


def _as_model_input(
    array: NDArray[np.floating],
    expected_shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float32]:
    """
    Convert one sequence array into a contiguous float32 batch of one.

    The source array is only ever read, so a caller's BeatSequence is
    never modified, whatever dtype or memory layout it arrives in.
    """

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} input must have shape {expected_shape}, found {array.shape}."
        )

    # Converts its memory to a continuous block and add a new axis of 1 at
    # the beginning representing the batch_size
    return np.ascontiguousarray(array, dtype=np.float32)[np.newaxis, ...]


def _validate_logits(raw_logits: object) -> NDArray[np.float32]:
    """Check the model returned exactly one four-class prediction."""

    # Convert to np array
    logits = np.asarray(raw_logits, dtype=np.float32)

    # Check we have recieved the expected number of logits
    if logits.shape != (1, NUM_CLASSES):
        raise ValueError(
            f"ONNX output {ONNX_OUTPUT_NAME!r} must have shape "
            f"(1, {NUM_CLASSES}) for a single sequence, found {logits.shape}."
        )

    # Drop the batch axis and freeze it, so a caller cannot edit the
    # logits held by the PredictionEvent. Changes the shape from
    # (1, NUM_CLASSES) to (NUM_CLASSES,)
    single = logits[0].copy()
    single.setflags(write=False)

    return single
