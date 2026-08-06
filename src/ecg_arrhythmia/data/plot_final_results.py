import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecg_arrhythmia.plotting import (
    MODEL_EVALUATION_FIGURES_DIR
)


def load_metrics(metrics_path: Path) -> dict:
    """
    Load one saved metrics JSON file.
    """

    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file found at {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_confusion_matrix(summary: dict) -> tuple[np.ndarray, list[str]]:
    """
    Convert the saved JSON confusion matrix into a 2D numpy array.
    """

    labels = summary["confusion_matrix"]["labels"]
    rows = summary["confusion_matrix"]["rows"]

    matrix = np.array(
        [[row["predictions"][label] for label in labels] for row in rows],
        dtype=int,
    )

    return matrix, labels


def row_normalise(matrix: np.ndarray) -> np.ndarray:
    """
    Turn the confusion matrix counts into row-wise proportions.
    """

    row_sums = matrix.sum(axis=1, keepdims=True)

    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )


def plot_overall_metrics(
    cnn_summary: dict,
    transformer_summary: dict,
    output_path: Path,
) -> None:
    """
    Plot accuracy and macro F1 for the final CNN and Transformer.
    """

    # Gives the actual keys used to index the loaded dict
    metric_keys = ["accuracy", "macro_f1"]
    # Gives the metrics names we will use on the plot
    metric_labels = ["Accuracy", "Macro F1"]

    # Grab the accuracy and macro from each summary
    cnn_values = [cnn_summary[key] for key in metric_keys]
    transformer_values = [transformer_summary[key] for key in metric_keys]

    # Gives us [0, 1].
    x = np.arange(len(metric_labels))
    # Total width of each bar
    width = 0.35

    # Define the canvas width
    plt.figure(figsize=(8, 5))

    cnn_bars = plt.bar(
        # Performs this for on [0, 1]
        x - (width / 2),  # Shift width / 2 to the left of x
        cnn_values,  # plots the accuracy and macro at these values
        width,
        label="CNN V2 + RR",
        color="tab:blue",
        edgecolor="black",
        linewidth=1,
    )

    transformer_bars = plt.bar(
        x + (width / 2),  # Shift width / 2 to the right of x
        transformer_values,  # plots the accuracy and macro at these values
        width,
        label="Transformer",
        color="tab:red",
        edgecolor="black",
        linewidth=1,
    )

    # Plots the actual values above the bar. padding controls
    # how far above the bar they are placed.
    plt.bar_label(cnn_bars, fmt="%.4f", padding=3)
    plt.bar_label(transformer_bars, fmt="%.4f", padding=3)

    # xitcks places the labels at positions in x
    plt.xticks(x, metric_labels)
    # How high the y - axis is. 1.05 gives a bit more room
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title("Final Target-Matched Test Metrics")
    # Displays the label we gave in each plt.bar
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(
    cnn_summary: dict,
    transformer_summary: dict,
    output_path: Path,
) -> None:
    """
    Plot final per-class F1 for the CNN and Transformer.
    """

    class_labels = ["N", "S", "V", "F"]

    cnn_values = [cnn_summary["per_class"][label]["f1"] for label in class_labels]
    transformer_values = [
        transformer_summary["per_class"][label]["f1"] for label in class_labels
    ]

    x = np.arange(len(class_labels))
    width = 0.35

    plt.figure(figsize=(8, 5))

    cnn_bars = plt.bar(
        x - width / 2,
        cnn_values,
        width,
        label="CNN V2 + RR",
        color="tab:blue",
        edgecolor="black",
        linewidth=1,
    )

    transformer_bars = plt.bar(
        x + width / 2,
        transformer_values,
        width,
        label="Transformer",
        color="tab:red",
        edgecolor="black",
        linewidth=1,
    )

    plt.bar_label(cnn_bars, fmt="%.4f", padding=3)
    plt.bar_label(transformer_bars, fmt="%.4f", padding=3)

    plt.xticks(x, class_labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("F1 Score")
    plt.title("Target-Matched Per-Class Test F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    summary: dict,
    title: str,
    output_path: Path,
    cmap: str,
) -> None:
    """
    Plot a confusion matrix with raw counts and row-normalised percentages.
    """

    matrix, labels = extract_confusion_matrix(summary)

    plot_row_normalised_matrix(
        matrix=matrix,
        labels=labels,
        title=title,
        output_path=output_path,
        cmap=cmap,
        x_label="Predicted label",
        y_label="True label",
    )


def main() -> None:
    # Create the directory we're going to save our figures to.
    figures_dir = MODEL_EVALUATION_FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    # define where our results are kept
    cnn_metrics_path = Path(
        "artifacts/results/cnn_baseline_v2_rr_sequence_targets_test_metrics.json"
    )
    transformer_metrics_path = Path(
        "artifacts/results/ecg_sequence_transformer_tuned_matched_test_metrics.json"
    )

    # Extract them from the defined paths
    cnn_summary = load_metrics(cnn_metrics_path)
    transformer_summary = load_metrics(transformer_metrics_path)

    plot_overall_metrics(
        cnn_summary=cnn_summary,
        transformer_summary=transformer_summary,
        output_path=figures_dir / "cnn_vs_transformer_overall_metrics.png",
    )

    plot_per_class_f1(
        cnn_summary=cnn_summary,
        transformer_summary=transformer_summary,
        output_path=figures_dir / "cnn_vs_transformer_per_class_f1.png",
    )

    plot_confusion_matrix(
        summary=cnn_summary,
        title="CNN V2 + RR Confusion Matrix (Row-Normalised)",
        output_path=figures_dir / "cnn_v2_rr_target_matched_confusion_matrix.png",
        cmap="Blues",
    )

    plot_confusion_matrix(
        summary=transformer_summary,
        title="Transformer Confusion Matrix (Row-Normalised)",
        output_path=figures_dir / "transformer_tuned_confusion_matrix.png",
        cmap="Reds",
    )

    print("Saved final result figures to:", figures_dir)


if __name__ == "__main__":
    main()
