"""
Figures for the FP32 versus INT8 deployment benchmark.

Latency bars use the mean across repeats with error bars spanning the
repeat minimum and maximum, so run-to-run variation is visible in the
figure rather than hidden behind a single number.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

FP32_COLOUR = "steelblue"
INT8_COLOUR = "darkorange"


def write_benchmark_figures(
    fp32_summary: dict,
    int8_summary: dict,
    size: dict,
    per_record: list[dict],
    figures_dir: Path,
) -> list[Path]:
    """Save the benchmark comparison figures and return their paths."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    latency_path = figures_dir / "fp32_vs_int8_latency.png"
    plot_latency_comparison(fp32_summary, int8_summary, latency_path)
    written.append(latency_path)

    throughput_path = figures_dir / "fp32_vs_int8_throughput.png"
    plot_throughput_comparison(fp32_summary, int8_summary, throughput_path)
    written.append(throughput_path)

    size_path = figures_dir / "fp32_vs_int8_model_size.png"
    plot_size_comparison(size, size_path)
    written.append(size_path)

    per_record_path = figures_dir / "per_record_mean_latency.png"
    plot_per_record_latency(per_record, per_record_path)
    written.append(per_record_path)

    return written


def _bars_with_spread(
    ax: plt.Axes,
    metrics: list[str],
    fp32_summary: dict,
    int8_summary: dict,
) -> None:
    """Grouped bars of across-repeat means with min-to-max error bars."""

    positions = np.arange(len(metrics))
    width = 0.38

    for offset, (label, summary, colour) in (
        (-width / 2, ("FP32", fp32_summary, FP32_COLOUR)),
        (width / 2, ("INT8", int8_summary, INT8_COLOUR)),
    ):
        means = [summary[metric]["mean"] for metric in metrics]
        lower = [
            summary[metric]["mean"] - summary[metric]["minimum"] for metric in metrics
        ]
        upper = [
            summary[metric]["maximum"] - summary[metric]["mean"] for metric in metrics
        ]

        bars = ax.bar(
            positions + offset,
            means,
            width,
            yerr=[lower, upper],
            capsize=4,
            label=label,
            color=colour,
            edgecolor="black",
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(positions)


def plot_latency_comparison(
    fp32_summary: dict,
    int8_summary: dict,
    output_path: Path,
) -> None:
    """Mean, median and p95 latency with run-to-run spread."""

    metrics = ["mean", "median", "p95"]

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()

    _bars_with_spread(ax, metrics, fp32_summary, int8_summary)

    ax.set_xticklabels(["Mean", "Median", "p95"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FP32 vs INT8 inference latency (mean across repeats, min-max spread)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_throughput_comparison(
    fp32_summary: dict,
    int8_summary: dict,
    output_path: Path,
) -> None:
    """Throughput with run-to-run spread."""

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()

    _bars_with_spread(ax, ["throughput"], fp32_summary, int8_summary)

    ax.set_xticklabels(["Throughput"])
    ax.set_ylabel("Sequences per second")
    ax.set_title("FP32 vs INT8 throughput (mean across repeats, min-max spread)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_size_comparison(size: dict, output_path: Path) -> None:
    """Model file sizes with the reduction annotated."""

    plt.figure(figsize=(6, 4.5))
    ax = plt.gca()

    bars = ax.bar(
        ["FP32", "INT8"],
        [size["fp32_mib"], size["int8_mib"]],
        color=[FP32_COLOUR, INT8_COLOUR],
        edgecolor="black",
    )
    ax.bar_label(bars, fmt="%.2f MiB", padding=3)

    ax.set_ylabel("Model size (MiB)")
    ax.set_title(
        f"Model size: {size['reduction_percentage']:.2f}% reduction "
        f"({size['compression_ratio']:.2f}x)"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_record_latency(per_record: list[dict], output_path: Path) -> None:
    """Mean latency per record for both models."""

    record_names = [record["record_name"] for record in per_record]
    positions = np.arange(len(record_names))
    width = 0.38

    plt.figure(figsize=(9, 4.5))
    ax = plt.gca()

    for offset, (label, key, colour) in (
        (-width / 2, ("FP32", "fp32", FP32_COLOUR)),
        (width / 2, ("INT8", "int8", INT8_COLOUR)),
    ):
        values = [record[key]["mean"] for record in per_record]
        bars = ax.bar(
            positions + offset,
            values,
            width,
            label=label,
            color=colour,
            edgecolor="black",
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(positions)
    ax.set_xticklabels(record_names)
    ax.set_xlabel("Validation record")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Per-record mean inference latency (pooled across repeats)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
