from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

MODEL_EVALUATION_FIGURES_DIR = Path("artifacts/figures/model_evaluation")

# The colour scale is fixed to the full percentage range so matrices are
# directly comparable and a rare class in full agreement is as dark as a
# common one.
PERCENTAGE_MINIMUM = 0.0
PERCENTAGE_MAXIMUM = 100.0


def row_normalise(matrix: NDArray) -> NDArray[np.float64]:
    """
    Turn matrix counts into row-wise proportions.

    An empty row stays at zero rather than producing NaN, so a class that
    never appears cannot poison the colour scale.
    """

    matrix = np.asarray(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)

    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )


def plot_row_normalised_matrix(
    matrix: NDArray,
    labels: list[str],
    title: str,
    output_path: Path,
    cmap: str,
    x_label: str,
    y_label: str,
    colorbar_label: str = "Percentage",
) -> None:
    """
    Plot a square matrix coloured by row percentage.

    Colouring by row rather than by raw count is what keeps a rare class
    readable: a row of five predictions that all agree is as dark as a
    row of two thousand that all agree. Each cell still shows its raw
    count above the row percentage, so nothing is hidden by the
    normalisation.
    """

    matrix = np.asarray(matrix)
    percentage_matrix = row_normalise(matrix) * 100

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    image = ax.imshow(
        percentage_matrix,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        vmin=PERCENTAGE_MINIMUM,
        vmax=PERCENTAGE_MAXIMUM,
    )

    plt.colorbar(image, label=colorbar_label)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Add thick black borders around every cell.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Use white text on darker cells so it stays readable.
    threshold = (PERCENTAGE_MINIMUM + PERCENTAGE_MAXIMUM) / 2.0

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            count = matrix[row_index, column_index]
            percentage = percentage_matrix[row_index, column_index]

            cell_text = f"{count}\n({percentage:.1f}%)"

            ax.text(
                column_index,
                row_index,
                cell_text,
                ha="center",
                va="center",
                color="white" if percentage > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
