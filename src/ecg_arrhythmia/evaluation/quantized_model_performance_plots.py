import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecg_arrhythmia.data.label_mapping import CLASS_LABELS
from ecg_arrhythmia.visualisation.matrix_plots import plot_row_normalised_matrix

logger = logging.getLogger(__name__)

CONFUSION_COLOUR_MAP = "Blues"


def write_performance_figures(
    fp32_metrics: dict,
    int8_metrics: dict,
    deltas: dict,
    outcomes: dict,
    figures_dir: Path,
) -> list[Path]:
    """Save the aggregate comparison figures and return their paths."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, metrics in (("fp32", fp32_metrics), ("int8", int8_metrics)):
        matrix_path = figures_dir / f"{name}_confusion_matrix.png"
        plot_row_normalised_matrix(
            matrix=np.asarray(metrics["confusion_matrix"], dtype=np.int64),
            labels=list(CLASS_LABELS),
            title=(f"Validation records: ground truth vs {name.upper()} predictions"),
            output_path=matrix_path,
            cmap=CONFUSION_COLOUR_MAP,
            x_label="Predicted label",
            y_label="True label",
        )
        written.append(matrix_path)

    for metric in ("f1", "recall"):
        metric_path = figures_dir / f"per_class_{metric}_comparison.png"
        plot_per_class_comparison(
            fp32_per_class=fp32_metrics["per_class"],
            int8_per_class=int8_metrics["per_class"],
            metric=metric,
            output_path=metric_path,
        )
        written.append(metric_path)

    delta_path = figures_dir / "metric_deltas.png"
    plot_metric_deltas(deltas=deltas, output_path=delta_path)
    written.append(delta_path)

    outcomes_path = figures_dir / "changed_prediction_outcomes.png"
    plot_outcome_counts(outcomes=outcomes, output_path=outcomes_path)
    written.append(outcomes_path)

    return written


def plot_per_class_comparison(
    fp32_per_class: dict,
    int8_per_class: dict,
    metric: str,
    output_path: Path,
) -> None:
    """Grouped FP32 and INT8 bars for one per-class metric."""

    positions = np.arange(len(CLASS_LABELS))
    width = 0.38

    fp32_values = [fp32_per_class[label][metric] for label in CLASS_LABELS]
    int8_values = [int8_per_class[label][metric] for label in CLASS_LABELS]

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()

    fp32_bars = ax.bar(
        positions - width / 2,
        fp32_values,
        width,
        label="FP32",
        color="steelblue",
        edgecolor="black",
    )
    int8_bars = ax.bar(
        positions + width / 2,
        int8_values,
        width,
        label="INT8",
        color="darkorange",
        edgecolor="black",
    )

    ax.bar_label(fp32_bars, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(int8_bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Ground-truth class")
    ax.set_ylabel(metric.capitalize() if metric != "f1" else "F1")
    ax.set_title(
        f"Per-class {metric.upper() if metric == 'f1' else metric}: FP32 vs INT8"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_deltas(deltas: dict, output_path: Path) -> None:
    """Signed INT8-minus-FP32 metric changes around a zero line."""

    names = ["Accuracy", "Macro F1"] + [f"{label} F1" for label in CLASS_LABELS]
    values = [deltas["accuracy_delta"], deltas["macro_f1_delta"]] + [
        deltas["per_class_deltas"][label]["f1_delta"] for label in CLASS_LABELS
    ]

    colours = ["darkorange" if value < 0 else "steelblue" for value in values]

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()

    bars = ax.bar(names, values, color=colours, edgecolor="black")
    ax.bar_label(bars, fmt="%+.4f", padding=3, fontsize=8)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("INT8 - FP32")
    ax.set_title("Metric change from INT8 quantisation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_outcome_counts(outcomes: dict, output_path: Path) -> None:
    """Counts of changed labelled predictions by ground-truth outcome."""

    names = [
        "FP32 correct\nINT8 wrong",
        "FP32 wrong\nINT8 correct",
        "Both wrong",
    ]
    values = [
        outcomes["fp32_correct_int8_wrong"],
        outcomes["fp32_wrong_int8_correct"],
        outcomes["both_wrong"],
    ]
    colours = ["darkorange", "steelblue", "grey"]

    plt.figure(figsize=(7, 4.5))
    ax = plt.gca()

    bars = ax.bar(names, values, color=colours, edgecolor="black")
    ax.bar_label(bars, padding=3)

    ax.set_ylabel("Changed labelled predictions")
    ax.set_title("Ground-truth outcome of FP32-INT8 disagreements")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
