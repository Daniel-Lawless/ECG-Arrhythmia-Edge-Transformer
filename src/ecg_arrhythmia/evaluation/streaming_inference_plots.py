import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.data.label_mapping import CLASS_LABELS
from ecg_arrhythmia.visualisation.matrix_plots import plot_row_normalised_matrix

logger = logging.getLogger(__name__)

# Readable names for the two implementations in each comparison.
COMPARISON_TITLES = {
    "pytorch_vs_offline_onnx": ("PyTorch", "Offline ONNX"),
    "offline_onnx_vs_streaming_onnx": ("Offline ONNX", "Streaming ONNX"),
    "pytorch_vs_streaming_onnx": ("PyTorch", "Streaming ONNX"),
}

COMPARISON_CMAPS = {
    "pytorch_vs_offline_onnx": "Blues",
    "offline_onnx_vs_streaming_onnx": "Greens",
    "pytorch_vs_streaming_onnx": "Purples",
}

def write_aggregate_agreement_figures(
    comparisons: dict,
    figures_dir: Path,
) -> list[Path]:
    """Save aggregate agreement matrices across all evaluated records."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, comparison in comparisons.items():
        reference_label, comparison_label = COMPARISON_TITLES[name]

        output_path = figures_dir / f"aggregate_{name}_agreement.png"

        plot_prediction_agreement_matrix(
            matrix=np.asarray(
                comparison["agreement_matrix"],
                dtype=np.int64,
            ),
            reference_label=reference_label,
            comparison_label=comparison_label,
            comparison_name=name,
            title=(
                f"All validation records: {reference_label} vs "
                f"{comparison_label} prediction agreement"
            ),
            output_path=output_path,
        )

        written.append(output_path)

    return written


def write_record_figures(
    record_name: str,
    pytorch_logits: NDArray[np.float32],
    streaming_onnx_logits: NDArray[np.float32],
    target_peaks: NDArray[np.int64],
    comparisons: dict,
    figures_dir: Path,
) -> list[Path]:
    """Save every parity figure for one record and return their paths."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, comparison in comparisons.items():
        reference_label, comparison_label = COMPARISON_TITLES[name]
        output_path = figures_dir / f"record_{record_name}_{name}_agreement.png"

        plot_prediction_agreement_matrix(
            matrix=np.asarray(comparison["agreement_matrix"], dtype=np.int64),
            reference_label=reference_label,
            comparison_label=comparison_label,
            comparison_name=name,
            title=(
                f"Record {record_name}: {reference_label} vs "
                f"{comparison_label} prediction agreement"
            ),
            output_path=output_path,
        )
        written.append(output_path)

    maximum_differences = per_sequence_maximum_difference(
        pytorch_logits,
        streaming_onnx_logits,
    )

    scatter_path = figures_dir / f"record_{record_name}_logit_scatter.png"
    plot_logit_scatter(
        pytorch_logits=pytorch_logits,
        streaming_onnx_logits=streaming_onnx_logits,
        title=f"Record {record_name}: PyTorch vs streaming ONNX logits",
        output_path=scatter_path,
    )
    written.append(scatter_path)

    histogram_path = figures_dir / f"record_{record_name}_difference_histogram.png"
    plot_difference_histogram(
        maximum_differences=maximum_differences,
        title=(f"Record {record_name}: per-sequence maximum absolute logit difference"),
        output_path=histogram_path,
    )
    written.append(histogram_path)

    drift_path = figures_dir / f"record_{record_name}_difference_across_record.png"
    plot_difference_across_record(
        target_peaks=target_peaks,
        maximum_differences=maximum_differences,
        title=f"Record {record_name}: logit difference across the record",
        output_path=drift_path,
    )
    written.append(drift_path)

    return written


def per_sequence_maximum_difference(
    reference: NDArray[np.float32],
    comparison: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Largest absolute logit difference within each sequence."""

    reference = np.asarray(reference, dtype=np.float32)
    comparison = np.asarray(comparison, dtype=np.float32)

    if reference.size == 0:
        return np.zeros(0, dtype=np.float32)

    return np.abs(reference - comparison).max(axis=1)


def plot_prediction_agreement_matrix(
    matrix: NDArray[np.int64],
    reference_label: str,
    comparison_label: str,
    comparison_name: str,
    title: str,
    output_path: Path,
) -> None:

    plot_row_normalised_matrix(
        matrix=matrix,
        labels=list(CLASS_LABELS),
        title=title,
        output_path=output_path,
        cmap=COMPARISON_CMAPS[comparison_name],
        x_label=f"{comparison_label} prediction",
        y_label=f"{reference_label} prediction",
        colorbar_label="Row percentage",
    )


def plot_logit_scatter(
    pytorch_logits: NDArray[np.float32],
    streaming_onnx_logits: NDArray[np.float32],
    title: str,
    output_path: Path,
) -> None:
    """
    Plot every class logit from both paths against the ideal y = x.

    The axes are shared and equal so the line is a true diagonal and a
    real disagreement would visibly leave it.
    """

    reference = np.asarray(pytorch_logits, dtype=np.float32).ravel()
    comparison = np.asarray(streaming_onnx_logits, dtype=np.float32).ravel()

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    ax.scatter(reference, comparison, s=8, alpha=0.4, edgecolors="none")

    if reference.size > 0:
        lower = float(min(reference.min(), comparison.min()))
        upper = float(max(reference.max(), comparison.max()))
        padding = max((upper - lower) * 0.05, 1e-6)
        limits = (lower - padding, upper + padding)

        ax.plot(limits, limits, color="red", linewidth=1.0, label="y = x")
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        maximum_difference = float(np.abs(reference - comparison).max())
        ax.text(
            0.05,
            0.95,
            f"max |difference| = {maximum_difference:.3e}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PyTorch logit")
    ax.set_ylabel("Streaming ONNX logit")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_difference_histogram(
    maximum_differences: NDArray[np.float32],
    title: str,
    output_path: Path,
) -> None:
    """Plot the distribution of per-sequence maximum absolute difference."""

    differences = np.asarray(maximum_differences, dtype=np.float64)

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()

    if differences.size > 0 and differences.max() > 0:
        ax.hist(differences, bins=40, color="steelblue", edgecolor="black")
    else:
        # Every sequence matched exactly, so a single bar at zero is the
        # honest picture rather than an empty axis.
        ax.bar([0.0], [differences.size], width=1.0, color="steelblue")
        ax.set_xlim(-1.0, 1.0)

    ax.set_xlabel("Maximum absolute logit difference")
    ax.set_ylabel("Sequences")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_difference_across_record(
    target_peaks: NDArray[np.int64],
    maximum_differences: NDArray[np.float32],
    title: str,
    output_path: Path,
) -> None:
    """
    Plot difference against position in the record.

    A rising trend would indicate drift or state leaking between
    sequences; a flat line at zero is what parity looks like.
    """

    peaks = np.asarray(target_peaks, dtype=np.int64)
    differences = np.asarray(maximum_differences, dtype=np.float64)

    plt.figure(figsize=(10, 4))
    ax = plt.gca()

    ax.plot(peaks, differences, linewidth=0.8, marker="o", markersize=2)

    ax.set_xlabel("Target R-peak sample index")
    ax.set_ylabel("Maximum absolute logit difference")
    ax.set_title(title)

    if differences.size > 0 and differences.max() == 0:
        ax.set_ylim(-1.0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
