import argparse
import logging
from pathlib import Path

import torch

from ecg_arrhythmia.data.label_mapping import NUM_CLASSES
from ecg_arrhythmia.models.sequence_transformer import ECGSequenceTransformer
from ecg_arrhythmia.preprocessing.beat_extraction import (
    RR_FEATURE_DIM,
    WINDOW_SIZE,
)

logger = logging.getLogger(__name__)

# A singleton example dimension may be seen as fixed during
# torch.export. Using 2 provides a non-singleton example while
# dynamic_shapes explicitly declares the batch dimension as dynamic.
EXAMPLE_BATCH_SIZE = 2


def load_model(
    checkpoint_path: Path,
    num_layers: int,
) -> ECGSequenceTransformer:

    logger.info("Attempting to load model...")

    device = torch.device("cpu")

    # Checks if the file path exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    logger.info("Loading saved checkpoint...")

    # Load the weights
    saved_state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    logger.info("Checkpoint successfully loaded")

    # Create an empty model skeleton. Must use the same architecture
    # that was used to create the checkpoint during training
    model = ECGSequenceTransformer(num_layers=num_layers, num_classes=NUM_CLASSES).to(
        device
    )

    # Load the saved weights into the models skeleton
    model.load_state_dict(saved_state_dict)

    logger.info("Saved checkpoint loaded into model")

    # Switch the model to evaluation behaviour so
    # dropout is disabled and BatchNorm uses its learned running statistics.
    # Gradient tracking is disabled separately during export using torch.no_grad().
    model.eval()

    logger.info("Model successfully loaded.")

    return model


def export_model(
    model: ECGSequenceTransformer,
    output_path: Path,
    sequence_length: int,
) -> None:
    """
    Export our trained PyTorch transformer to ONNX

    The sequence length is fixed, but batch_size is dynamic
    """

    logger.info("Attempting to export model...")

    if sequence_length <= 0:
        raise ValueError(
            f"Sequence length cannot be 0 or negative. Got {sequence_length}"
        )

    # Create the output path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # These tensors show the exporter what one valid model call looks like.
    #
    # ECG shape:
    # (batch_size, sequence_length, channel, window_size)
    example_ecg = torch.randn(
        EXAMPLE_BATCH_SIZE, sequence_length, 1, WINDOW_SIZE, dtype=torch.float32
    )
    # This fill a tensor of shape
    # (EXAMPLE_BATCH_SIZE, sequence_length, 1, WINDOW_SIZE)
    # with values randomly chosen from a standard normal distribution

    # RR shape:
    # (batch_size, sequence_length, rr_feature_dim)
    example_rr = torch.randn(
        EXAMPLE_BATCH_SIZE, sequence_length, RR_FEATURE_DIM, dtype=torch.float32
    )

    # Both model inputs must have the same batch_size
    # torch.export.Dim if used to define a dynamic dimension
    # for exporting a PyTorch model, min=1 means that
    # although this dimension is dynamic, it will never < 1
    batch_size = torch.export.Dim("batch_size", min=1)

    # The keys represent the inputs going into our PyTorch model. They are
    # the parameter names our forward pass accepts.
    # it says that the 0th dimension of each input can be dynamic
    dynamic_shapes = {"x": {0: batch_size}, "rr": {0: batch_size}}

    logger.info("Creating onnx computational graph...")

    """
    Exports the PyTorch model to the ONNX format for deployment.
    It uses the provided dummy tensors (example_ecg, example_rr) 
    to trace the computational graph, applies the dynamic shape
    rules to allow for variable batch sizes at runtime, 
    and maps clear string names to the inputs and outputs
    via the TorchDynamo engine.

    Tracing bridges dynamic PyTorch with static ONNX. It uses 
    the dummy data to record a hardcoded map of every mathematical
    operation in the forward pass. torch.no_grad() ensures we don't
    track gradients during this
    """
    with torch.no_grad():
        onnx_program = torch.onnx.export(
            model=model,
            args=(example_ecg, example_rr),
            input_names=["ecg_sequence", "rr_sequence"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )

    logger.info("Computational graph successfully created")

    # Lets Pylance know onnxprogram is not None at this point
    assert onnx_program is not None

    # Save exported model
    onnx_program.save(output_path)

    logger.info("Exported ONNX model to: %s", output_path)


def parse_args() -> argparse.Namespace:

    # Define parser
    parser = argparse.ArgumentParser(description="CLI for exporting model")

    # Points to where the model checkpoint is saved.
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/models/ecg_sequence_transformer_tuned.pt"),
        help="Where the models checkpoint lives",
    )

    # Points to where the exported model should be saves
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/models/ecg_sequence_transformer.onnx"),
        help="Path where the exported ONNX model will be saved.",
    )

    # Number of layer in our transformer
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of layers our transformer used",
    )

    # Number of beats per sequence
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Length of each sequence in the transformer",
    )

    return parser.parse_args()


def main():
    # Extract the cl arguments
    args = parse_args()

    # Load the model
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        num_layers=args.num_layers,
    )

    # Export the model
    export_model(
        model=model,
        output_path=args.output_path,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
