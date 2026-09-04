import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

NATIVE_COLOUR = "steelblue"
DOCKER_COLOUR = "darkorange"
REFERENCE_COLOUR = "red"

DEFAULT_RESULTS_DIR = Path("artifacts/results/deployment_evaluation/docker_vs_native")
DEFAULT_FIGURES_DIR = Path("artifacts/figures/deployment_evaluation/docker_vs_native")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return result


def _label_points(axis, values: list[float], decimals: int = 2) -> None:
    span = max(values) - min(values)
    offset = span * 0.08 if span else max(abs(values[0]) * 0.015, 0.5)

    for index, value in enumerate(values):
        axis.text(
            index,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_paced_latency(summary: dict, output_path: Path) -> Path:
    """Compare the saved complete-path Native and Docker percentiles."""

    metrics = (("mean", "Mean"), ("p95", "p95"), ("p99", "p99"))
    native = summary["native"]["paced"]["full_path_ms"]
    docker = summary["docker"]["paced"]["full_path_ms"]
    deadline_ms = float(summary["methodology"]["deadline_ms"])

    native_values = [float(native[key]) for key, _ in metrics]
    docker_values = [float(docker[key]) for key, _ in metrics]
    positions = np.arange(len(metrics))
    width = 0.34

    figure, axis = plt.subplots(figsize=(9, 5.4))
    native_bars = axis.bar(
        positions - width / 2,
        native_values,
        width,
        label="Native",
        color=NATIVE_COLOUR,
        edgecolor="0.25",
        linewidth=0.8,
    )
    docker_bars = axis.bar(
        positions + width / 2,
        docker_values,
        width,
        label="Docker",
        color=DOCKER_COLOUR,
        edgecolor="0.25",
        linewidth=0.8,
    )
    axis.axhline(
        deadline_ms,
        color=REFERENCE_COLOUR,
        linestyle="--",
        linewidth=1.2,
        label=f"{deadline_ms:.0f} ms deadline",
    )

    axis.bar_label(native_bars, fmt="%.1f", padding=3, fontsize=9)
    axis.bar_label(docker_bars, fmt="%.1f", padding=3, fontsize=9)
    axis.set_xticks(positions, [label for _, label in metrics])
    axis.set_ylabel("Complete-path latency (ms)")
    axis.set_ylim(0, deadline_ms * 1.1)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", ncols=3)
    axis.set_title("Native vs Docker paced streaming latency on Raspberry Pi 5")

    changes = summary["docker_change_percent"]["paced_full_path_ms"]
    validation = summary["validation"]
    docker_margin = summary["docker"]["paced"]["minimum_deadline_margin_ms"]
    figure.text(
        0.5,
        0.015,
        (
            f"Docker change: mean {changes['mean']:+.1f}%  •  "
            f"p95 {changes['p95']:+.1f}%  •  p99 {changes['p99']:+.1f}%\n"
            f"{validation['total_primary_chunks']:,} chunks  •  "
            f"{validation['deadline_misses']} deadline misses  •  "
            f"worst Docker headroom {docker_margin:.3f} ms"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    figure.tight_layout(rect=(0, 0.10, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path


def _summary_panel(
    axis,
    labels: list[str],
    values: list[float],
    y_label: str,
    decimals: int = 2,
) -> None:
    positions = np.arange(len(labels))
    axis.scatter(
        positions,
        values,
        color=DOCKER_COLOUR,
        s=36,
        zorder=2,
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel(y_label, fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    axis.margins(x=0.16, y=0.22)
    _label_points(axis, values, decimals=decimals)


def plot_sustained_resources(sustained: dict, output_path: Path) -> Path:
    """Plot the run-wide resource statistics retained in ``sustained.json``.

    The curated result intentionally excludes timestamped telemetry samples.
    Consequently this figure presents measured run summaries rather than
    inventing a time series from aggregate values.
    """

    resources = sustained["resources"]
    duration_minutes = float(sustained["duration"]["measured_wall_seconds"]) / 60

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    _summary_panel(
        axes[0, 0],
        ["Mean", "Maximum"],
        [
            float(resources["temperature_c"]["mean"]),
            float(resources["temperature_c"]["maximum"]),
        ],
        "Temperature (°C)",
    )
    _summary_panel(
        axes[0, 1],
        ["Mean", "Maximum"],
        [
            float(resources["cpu_frequency_mhz"]["mean"]),
            float(resources["cpu_frequency_mhz"]["maximum"]),
        ],
        "CPU frequency (MHz)",
        decimals=0,
    )
    _summary_panel(
        axes[1, 0],
        ["Start", "Mean", "End", "Maximum"],
        [
            float(resources["rss_mib"]["start"]),
            float(resources["rss_mib"]["mean"]),
            float(resources["rss_mib"]["end"]),
            float(resources["rss_mib"]["maximum"]),
        ],
        "Process RSS (MiB)",
    )
    _summary_panel(
        axes[1, 1],
        ["Mean", "Maximum"],
        [
            float(resources["process_cpu_percent"]["mean"]),
            float(resources["process_cpu_percent"]["maximum"]),
        ],
        "Process CPU (% of one core)",
    )

    figure.suptitle(
        "Docker sustained deployment: 60-minute resource summary",
        y=0.99,
    )
    figure.text(
        0.5,
        0.935,
        ("Run-wide statistics; timestamped telemetry samples were not retained"),
        ha="center",
        color="0.35",
        fontsize=9,
    )

    metrics = sustained["metrics"]
    throttling = resources["throttling"]
    figure.text(
        0.5,
        0.015,
        (
            f"{duration_minutes:.1f} minutes  •  "
            f"{metrics['chunks_processed']:,} chunks  •  "
            f"{metrics['deadline']['deadline_misses']} deadline misses  •  "
            "minimum headroom "
            f"{metrics['deadline']['minimum_deadline_margin_ms']:.3f} ms  •  "
            f"{throttling['reading_count'] - throttling['active_count']}/"
            f"{throttling['reading_count']} clear throttling readings"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    figure.tight_layout(rect=(0, 0.07, 1, 0.91), h_pad=2.0, w_pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path


def write_docker_benchmark_figures(
    summary_path: Path,
    sustained_path: Path,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Create the two figures supported by the curated benchmark outputs."""

    summary = _load_json(summary_path)
    sustained = _load_json(sustained_path)

    return [
        plot_paced_latency(
            summary,
            figures_dir / "native_vs_docker_paced_latency.png",
        ),
        plot_sustained_resources(
            sustained,
            figures_dir / "docker_sustained_resources.png",
        ),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Plot the curated Raspberry Pi Native-vs-Docker benchmark."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "summary.json",
    )
    parser.add_argument(
        "--sustained-json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "sustained.json",
    )
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    for path in write_docker_benchmark_figures(
        args.summary_json,
        args.sustained_json,
        args.figures_dir,
    ):
        logger.info("Wrote figure %s", path)


if __name__ == "__main__":
    main()
