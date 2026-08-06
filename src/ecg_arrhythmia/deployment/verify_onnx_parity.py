import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.ecg_sequence_dataset import ECGSequenceDataset
from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer
from ecg_arrhythmia.training.transformer_training import NUM_CLASSES

logger = logging.getLogger(__name__)

ONNX_ECG_INPUT_NAME = "ecg_sequence"
ONNX_RR_INPUT_NAME = "rr_sequence"
ONNX_OUTPUT_NAME = "logits"


class ParitySummary(TypedDict):
    total_predictions: int
    matching_predictions: int
    prediction_agreement: float
    maximum_absolute_logit_difference: float
    mean_absolute_logit_difference: float
    relative_tolerance: float
    absolute_tolerance: float
    logits_within_tolerance: bool
    predictions_match: bool
    parity_passed: bool


def load_pytorch_model(
    checkpoint_path: Path,
    num_layers: int,
    dropout: float,
) -> ECGSequenceTransformer:
    """
    Recreate the tuned PyTorch model and load its saved weights.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No PyTorch checkpoint found at {checkpoint_path}")

    # Create model skeleton
    model = ECGSequenceTransformer(
        num_classes=NUM_CLASSES,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Load our saved weights
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Put them into our model
    model.load_state_dict(state_dict)

    # Put model in eval mode
    model.eval()

    return model


def validate_onnx_model(onnx_path: Path) -> None:
    """
    Check that the saved file contains a structurally valid ONNX model.
    """

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


def verify_parity(
    pytorch_model: ECGSequenceTransformer,
    onnx_session: ort.InferenceSession,
    test_loader: DataLoader,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> ParitySummary:
    """
    Compare PyTorch and ONNX logits across the test set.
    """

    max_absolute_difference = 0.0
    total_absolute_difference = 0.0
    total_logit_values = 0

    matching_predictions = 0
    total_predictions = 0
    all_batches_close = True

    with torch.inference_mode():
        # We don't care about the labels here
        for batch_index, (ecg_batch, rr_batch, _) in enumerate(test_loader):
            # Keep both implementations on CPU using float32 inputs.
            ecg_batch = ecg_batch.to(
                device="cpu",
                dtype=torch.float32,
            )
            rr_batch = rr_batch.to(
                device="cpu",
                dtype=torch.float32,
            )

            # Run the original PyTorch model.
            pytorch_logits: np.ndarray = pytorch_model(
                ecg_batch,
                rr_batch,
            ).numpy()

            # ONNX Runtime consumes NumPy arrays. This is the actual inference step.
            # contiguous() is used here since PyTorch sometimes stores tensors
            # in fragmented blocks of memory. ONNX runtime expects a contiguous
            # block of memory, so calling this force PyTorch to reallocate the tensor
            # into a continuous block before making it a numpy array.

            onnx_logits = onnx_session.run(
                output_names=[ONNX_OUTPUT_NAME],
                input_feed={
                    ONNX_ECG_INPUT_NAME: ecg_batch.contiguous().numpy(),
                    ONNX_RR_INPUT_NAME: rr_batch.contiguous().numpy(),
                },
            )[0]

            assert isinstance(onnx_logits, np.ndarray)

            # session.run() returns a list to support multiple outputs.
            # [0] extracts our single logits array. This array contains
            # the model's predictions for every datapoint in the batch.
            # The resulting shape is (batch_size, num_classes).
            if pytorch_logits.shape != onnx_logits.shape:
                raise ValueError(
                    "Output shape mismatch: "
                    f"PyTorch={pytorch_logits.shape}, "
                    f"ONNX={onnx_logits.shape}"
                )

            # Both should have shape (batch_size, num_classes),
            # so absolute difference should be the same size. These
            # calculates how much the logits deviate.
            absolute_difference = np.abs(pytorch_logits - onnx_logits)

            # calculate the max between the running absolute max
            # difference and the max difference this this batch
            max_absolute_difference = max(
                max_absolute_difference,
                float(absolute_difference.max()),
            )

            # Append the total difference in this batch to the overall total
            # difference
            total_absolute_difference += float(absolute_difference.sum())

            # .size returns all values, so batch_size * num classes
            # = total number of logit values fo this batch, then add
            # that to the running total logit values
            total_logit_values += absolute_difference.size

            # Validates if all elements in both arrays fall within the
            # acceptable error margins. Returns True if all absolute value
            # differences fall beneath the tolerance |a - b| <= atol + rtol * |b|
            # it uses a flat baseline (atol) and a proportional, scaling
            # threshold (rtol).
            batch_is_close = np.allclose(
                pytorch_logits,
                onnx_logits,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )

            # If any single difference falls below this threshold parity
            # has failed.
            if not batch_is_close:
                all_batches_close = False
                logger.error(
                    "Logit parity failed for batch %d",
                    batch_index,
                )

            # Returns the index of the logit with the largest value in each row
            pytorch_predictions = pytorch_logits.argmax(axis=1)
            onnx_predictions = onnx_logits.argmax(axis=1)

            # Update the running matching_predictions and total predicitons
            matching_predictions += int(np.sum(pytorch_predictions == onnx_predictions))
            total_predictions += len(pytorch_predictions)

    # If no predictions have been made, throw an error
    if total_predictions == 0:
        raise ValueError("Cannot verify parity on an empty dataset")

    # Gives the average difference per logit
    mean_absolute_difference = total_absolute_difference / total_logit_values

    # Calculates the proportion of identical class predictions between both models.
    prediction_agreement = matching_predictions / total_predictions

    logger.info(
        "Maximum absolute logit difference: %.10f",
        max_absolute_difference,
    )
    logger.info(
        "Mean absolute logit difference: %.10f",
        mean_absolute_difference,
    )
    logger.info(
        "Prediction agreement: %d/%d (%.4f%%)",
        matching_predictions,
        total_predictions,
        prediction_agreement * 100,
    )

    predictions_match = matching_predictions == total_predictions

    parity_summary: ParitySummary = {
        "total_predictions": total_predictions,
        "matching_predictions": matching_predictions,
        "prediction_agreement": prediction_agreement,
        "maximum_absolute_logit_difference": max_absolute_difference,
        "mean_absolute_logit_difference": mean_absolute_difference,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "logits_within_tolerance": all_batches_close,
        "predictions_match": predictions_match,
        "parity_passed": all_batches_close and predictions_match,
    }

    return parity_summary


def save_parity_summary(
    summary: ParitySummary,
    output_path: Path,
) -> None:
    """
    Save the PyTorch-ONNX parity results as JSON.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    logger.info("Saved parity summary to: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CLI to comfirm parity between tuned PyTorch model "
            "and its exported ONNX model"
        )
    )

    # CLI arguments
    parser.add_argument(
        "--test-split-dir",
        type=Path,
        default=Path("data/splits_sequences_matched/test"),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/results/pytorch_onnx_parity_summary.json"),
        help="Where to save the PyTorch-ONNX parity results.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/models/ecg_sequence_transformer_tuned.pt"),
    )

    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("artifacts/models/ecg_sequence_transformer.onnx"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1e-5,
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    logger.info("Loading matched sequence test set...")

    # Load test set into our Dataset object
    test_set = ECGSequenceDataset(args.test_split_dir)

    # Create the test loader
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    logger.info("Loading tuned PyTorch model...")

    pytorch_model = load_pytorch_model(
        checkpoint_path=args.checkpoint_path,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    logger.info("Validating ONNX model...")
    # Validate the path of the onnx model
    validate_onnx_model(args.onnx_path)

    logger.info("Creating ONNX Runtime session...")
    onnx_session = create_onnx_session(args.onnx_path)

    logger.info("Comparing PyTorch and ONNX outputs...")

    # Verify they make the same predictions
    parity_summary = verify_parity(
        pytorch_model=pytorch_model,
        onnx_session=onnx_session,
        test_loader=test_loader,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )

    # Save our parity summary
    save_parity_summary(
        summary=parity_summary,
        output_path=args.output_path,
    )

    # We raise an error after we have saved the results so we have
    # a chance to look at the metrics.
    #
    # Fails if even one batch fails the tolerance check,
    # if all predictions were not equal, or both.
    if not parity_summary["parity_passed"]:
        raise AssertionError(
            "PyTorch-ONNX parity verification failed. "
            f"See {args.output_path} for details."
        )

    logger.info("PyTorch-ONNX parity verification passed")


if __name__ == "__main__":
    main()
