import logging
from pathlib import Path

import onnxruntime as ort

from ecg_arrhythmia.data.label_mapping import NUM_CLASSES
from ecg_arrhythmia.preprocessing.beat_extraction import (
    RR_FEATURE_DIM,
    SEQUENCE_LENGTH,
    WINDOW_SIZE,
)

logger = logging.getLogger(__name__)

ONNX_ECG_INPUT_NAME = "ecg_sequence"
ONNX_RR_INPUT_NAME = "rr_sequence"
ONNX_OUTPUT_NAME = "logits"

# ONNX Runtime reports float32 tensors with this type string.
ONNX_FLOAT_TYPE = "tensor(float)"

# Expected model input ranks, batch dimension included.
ECG_INPUT_RANK = 4
RR_INPUT_RANK = 3
LOGITS_OUTPUT_RANK = 2


def validate_onnx_model(onnx_path: Path) -> None:
    """
    Check that the saved file contains a structurally valid ONNX model.

    The onnx package is imported here rather than at module level. It is
    a graph-manipulation library that running inference never needs, so a
    deployment that only creates sessions should not have to load it.
    """

    import onnx

    if not onnx_path.exists():
        raise FileNotFoundError(f"No ONNX model found at {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    logger.info("ONNX model passed structural validation")


def create_onnx_session(
    onnx_path: Path,
) -> ort.InferenceSession:
    """
    Load the ONNX graph using the CPU execution provider.
    """

    if not onnx_path.is_file():
        raise FileNotFoundError(f"No ONNX model found at {onnx_path}")

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    """
    These are the input and output names we specified in the export. For
    every distinct input and output node defined in your model's
    computational graph, ONNX creates a ArgNode object.
    session.get_inputs returns a list of ArgNode objects, 2 in our case
    since we defined 2 inputs, that we can iterate through and extract the
    name of the input (also, the shape or type if we want). Same for the outputs.
    """
    input_names = {model_input.name for model_input in session.get_inputs()}
    output_names = {model_output.name for model_output in session.get_outputs()}

    # Validate that the input and output names are what we expect
    expected_input_names = {
        ONNX_ECG_INPUT_NAME,
        ONNX_RR_INPUT_NAME,
    }

    if input_names != expected_input_names:
        raise ValueError(
            "Unexpected ONNX input names. "
            f"Expected {expected_input_names}, found {input_names}"
        )

    if ONNX_OUTPUT_NAME not in output_names:
        raise ValueError(
            f"Expected ONNX output named {ONNX_OUTPUT_NAME}. Found {output_names}"
        )

    logger.info("ONNX inputs: %s", input_names)
    logger.info("ONNX outputs: %s", output_names)

    return session


def validate_session_contract(session: ort.InferenceSession) -> None:
    """
    Check a loaded session against the beat and class contract.

    Only statically known dimensions are compared. The batch axis is
    dynamic by design. Names are already checked by create_onnx_session.
    """

    # Create a dictionary mapping each model input name to its ArgNode object.
    inputs = {model_input.name: model_input for model_input in session.get_inputs()}
    # gives the ecg argnode object
    ecg = inputs[ONNX_ECG_INPUT_NAME]
    # gives the rr argnode object
    rr = inputs[ONNX_RR_INPUT_NAME]

    # Ensures the number of dimensions are correct.
    _require_rank(ecg.name, ecg.shape, ECG_INPUT_RANK)
    _require_rank(rr.name, rr.shape, RR_INPUT_RANK)

    # Checks that the input types are tensor floats
    for model_input in (ecg, rr):
        if model_input.type != ONNX_FLOAT_TYPE:
            raise ValueError(
                f"ONNX input {model_input.name!r} must be "
                f"{ONNX_FLOAT_TYPE}, found {model_input.type!r}."
            )

    # Check the sequence length dimension is as expected
    _require_static_dimension(ecg.name, ecg.shape, 1, SEQUENCE_LENGTH)
    # Check the ECG signal dimension is as expected
    _require_static_dimension(ecg.name, ecg.shape, 2, 1)
    # Check the number of amplitude values are as expected
    _require_static_dimension(ecg.name, ecg.shape, 3, WINDOW_SIZE)
    # Check the sequence length dimension is as expected
    _require_static_dimension(rr.name, rr.shape, 1, SEQUENCE_LENGTH)
    # Check the number of rr features is as expected
    _require_static_dimension(rr.name, rr.shape, 2, RR_FEATURE_DIM)

    # Create a dictionary mapping the model output to its ArgNode object
    outputs = {
        model_output.name: model_output for model_output in session.get_outputs()
    }
    # set logits equal to its ArgNode object
    logits = outputs[ONNX_OUTPUT_NAME]

    # Check the output dimension is as expected
    _require_rank(logits.name, logits.shape, LOGITS_OUTPUT_RANK)
    # Check that the output gives us the expected number of classes.
    _require_static_dimension(logits.name, logits.shape, 1, NUM_CLASSES)

    logger.info(
        "Validated ONNX contract: %s%s, %s%s -> %s%s",
        ecg.name,
        tuple(ecg.shape),
        rr.name,
        tuple(rr.shape),
        logits.name,
        tuple(logits.shape),
    )


def _require_rank(name: str, shape: list, expected_rank: int) -> None:
    # If the number of dimensions do not match, raise an error
    if len(shape) != expected_rank:
        raise ValueError(
            f"ONNX tensor {name} must have rank {expected_rank}, found shape {shape}."
        )


def _require_static_dimension(
    name: str,
    shape: list,
    axis: int,
    expected: int,
) -> None:
    dimension = shape[axis]

    # A symbolic dimension is reported as a string and cannot be checked.
    if isinstance(dimension, int) and dimension != expected:
        raise ValueError(
            f"ONNX tensor {name} axis {axis} must be {expected}, found {dimension}."
        )
