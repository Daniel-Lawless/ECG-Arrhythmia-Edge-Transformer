"""
Figures for FP32 versus INT8 inference agreement.

Reuses the generic Section 3 drawing helpers rather than copying them:
the scatter, histogram and across-record plots are model-pair-agnostic,
and the agreement matrix uses the shared row-normalised implementation.
Oranges distinguishes these figures from the Section 3 parity plots at a
glance.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.data.label_mapping import CLASS_LABELS
from ecg_arrhythmia.evaluation.streaming_inference_plots import (
    per_sequence_maximum_difference,
    plot_difference_across_record,
    plot_difference_histogram,
    plot_logit_scatter,
)
from ecg_arrhythmia.visualisation.matrix_plots import plot_row_normalised_matrix

logger = logging.getLogger(__name__)

AGREEMENT_COLOUR_MAP = "Oranges"


def write_record_agreement_figures(
    record_name: str,
    fp32_logits: NDArray[np.float32],
    int8_logits: NDArray[np.float32],
    target_peaks: NDArray[np.int64],
    matrix: list[list[int]],
    fp32_margins: NDArray[np.float32],
    agreed: NDArray[np.bool_],
    figures_dir: Path,
) -> list[Path]:
    """Save every agreement figure for one record and return their paths."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    matrix_path = figures_dir / f"record_{record_name}_fp32_vs_int8_agreement.png"
    plot_row_normalised_matrix(
        matrix=np.asarray(matrix, dtype=np.int64),
        labels=list(CLASS_LABELS),
        title=f"Record {record_name}: FP32 vs INT8 prediction agreement",
        output_path=matrix_path,
        cmap=AGREEMENT_COLOUR_MAP,
        x_label="INT8 prediction",
        y_label="FP32 prediction",
        colorbar_label="Row percentage",
    )
    written.append(matrix_path)

    maximum_differences = per_sequence_maximum_difference(fp32_logits, int8_logits)

    scatter_path = figures_dir / f"record_{record_name}_logit_scatter.png"
    plot_logit_scatter(
        reference_logits=fp32_logits,
        comparison_logits=int8_logits,
        title=f"Record {record_name}: FP32 vs INT8 logits",
        output_path=scatter_path,
        x_label="FP32 logit",
        y_label="INT8 logit",
    )
    written.append(scatter_path)

    histogram_path = figures_dir / f"record_{record_name}_difference_histogram.png"
    plot_difference_histogram(
        maximum_differences=maximum_differences,
        title=(
            f"Record {record_name}: per-sequence maximum absolute "
            "FP32-INT8 logit difference"
        ),
        output_path=histogram_path,
    )
    written.append(histogram_path)

    drift_path = figures_dir / f"record_{record_name}_difference_across_record.png"
    plot_difference_across_record(
        target_peaks=target_peaks,
        maximum_differences=maximum_differences,
        title=f"Record {record_name}: FP32-INT8 drift across the record",
        output_path=drift_path,
    )
    written.append(drift_path)

    # Margins only mean something comparatively when both groups exist.
    if agreed.any() and (~agreed).any():
        margin_path = figures_dir / f"record_{record_name}_margin_comparison.png"
        plot_margin_comparison(
            agreeing_margins=fp32_margins[agreed],
            disagreeing_margins=fp32_margins[~agreed],
            title=(
                f"Record {record_name}: FP32 logit margin, "
                "agreeing vs disagreeing sequences"
            ),
            output_path=margin_path,
        )
        written.append(margin_path)

    return written


def write_aggregate_agreement_figures(
    matrix: list[list[int]],
    pooled_per_sequence_max: NDArray[np.float32],
    figures_dir: Path,
) -> list[Path]:
    """Save the aggregate agreement matrix and pooled drift histogram."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    matrix_path = figures_dir / "aggregate_fp32_vs_int8_agreement.png"
    plot_row_normalised_matrix(
        matrix=np.asarray(matrix, dtype=np.int64),
        labels=list(CLASS_LABELS),
        title="All validation records: FP32 vs INT8 prediction agreement",
        output_path=matrix_path,
        cmap=AGREEMENT_COLOUR_MAP,
        x_label="INT8 prediction",
        y_label="FP32 prediction",
        colorbar_label="Row percentage",
    )
    written.append(matrix_path)

    histogram_path = figures_dir / "aggregate_difference_histogram.png"
    plot_difference_histogram(
        maximum_differences=pooled_per_sequence_max,
        title=(
            "All validation records: per-sequence maximum absolute "
            "FP32-INT8 logit difference"
        ),
        output_path=histogram_path,
    )
    written.append(histogram_path)

    return written


def plot_margin_comparison(
    agreeing_margins: NDArray[np.float32],
    disagreeing_margins: NDArray[np.float32],
    title: str,
    output_path: Path,
) -> None:
    """
    Compare FP32 decision margins for agreeing and disagreeing sequences.

    If disagreements cluster at small margins, INT8 mainly changed
    predictions FP32 was already close to changing itself. Margins are
    raw logit gaps, not calibrated probabilities.
    """

    agreeing = np.asarray(agreeing_margins, dtype=np.float64)
    disagreeing = np.asarray(disagreeing_margins, dtype=np.float64)

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()

    ax.hist(
        agreeing,
        bins=40,
        density=True,
        alpha=0.6,
        color="steelblue",
        label=f"Agreeing ({agreeing.size})",
    )
    ax.hist(
        disagreeing,
        bins=40,
        density=True,
        alpha=0.7,
        color="darkorange",
        label=f"Disagreeing ({disagreeing.size})",
    )

    ax.set_xlabel("FP32 logit margin (winning - second highest)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
