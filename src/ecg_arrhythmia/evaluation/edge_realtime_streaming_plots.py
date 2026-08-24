import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

NANOSECONDS_PER_MILLISECOND = 1_000_000

FP32_COLOUR = "steelblue"
INT8_COLOUR = "darkorange"

DEFAULT_FIGURES_DIR = Path("artifacts/figures/edge_realtime_streaming")


def _load_paced(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {name: data[name] for name in data.files}


def _latency_ms(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return arrays["processing_ns"] / NANOSECONDS_PER_MILLISECOND


def _lateness_ms(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return (
        arrays["actual_start_ns"] - arrays["scheduled_ns"]
    ) / NANOSECONDS_PER_MILLISECOND


def _deadline_misses(arrays: dict[str, np.ndarray], chunk_period_ms: float) -> int:
    """Recompute misses from raw timestamps: completion > scheduled + period."""

    period_ns = round(chunk_period_ms * NANOSECONDS_PER_MILLISECOND)
    lateness = arrays["completion_ns"] - (arrays["scheduled_ns"] + period_ns)

    return int((lateness > 0).sum())


def _prefixed(figures_dir: Path, label: str, filename: str) -> Path:
    prefix = f"{label}_" if label else ""

    return figures_dir / f"{prefix}{filename}"


# ---------------------------------------------------------------------
#                    Single-Configuration Figures
# ---------------------------------------------------------------------


def plot_latency_distribution(
    fp32: dict[str, np.ndarray],
    int8: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Latency histogram with a log count axis.

    The near-zero population outnumbers the heavy detector-stride
    population by ~50:1, so a linear axis hides the entire story. Step
    outlines keep both models visible where the bars overlap.
    """

    fp32_ms = _latency_ms(fp32)
    int8_ms = _latency_ms(int8)
    bins = np.linspace(
        0.0,
        max(fp32_ms.max(), int8_ms.max()) * 1.05,
        80,
    )

    plt.figure(figsize=(9, 4.5))

    for label, values, colour in (
        ("FP32", fp32_ms, FP32_COLOUR),
        ("INT8", int8_ms, INT8_COLOUR),
    ):
        plt.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=colour,
        )

    plt.yscale("log")
    plt.xlabel("Chunk processing latency (ms)")
    plt.ylabel("Chunks (log scale)")
    plt.title("Paced chunk-processing latency distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _stacked_panels(
    fp32_values: np.ndarray,
    int8_values: np.ndarray,
    y_label: str,
    title: str,
    output_path: Path,
    reference_ms: float | None = None,
) -> None:
    """FP32 above INT8 with shared axes, so neither model hides the other."""

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6.5),
        sharex=True,
        sharey=True,
    )

    for axis, label, values, colour in (
        (axes[0], "FP32", fp32_values, FP32_COLOUR),
        (axes[1], "INT8", int8_values, INT8_COLOUR),
    ):
        axis.plot(values, linewidth=0.5, color=colour)

        if reference_ms is not None:
            axis.axhline(
                reference_ms,
                color="red",
                linestyle="--",
                linewidth=1,
                label="Chunk period",
            )
            axis.legend(loc="upper right")

        axis.set_ylabel(y_label)
        axis.set_title(label, fontsize=10)

    axes[1].set_xlabel("Chunk index")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_realtime_figures(
    fp32_npz_path: Path,
    int8_npz_path: Path,
    chunk_period_ms: float,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    label: str = "",
    include_latency_over_record: bool = True,
) -> list[Path]:
    """
    Render one configuration's paced figures and return their paths.

    A non-empty label (for example "performance" or "ondemand") is
    prefixed to every filename so two configurations never overwrite
    each other's figures. The latency-over-record view is skipped when
    the governor-comparison figure will already show the same panels.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    fp32 = _load_paced(fp32_npz_path)
    int8 = _load_paced(int8_npz_path)
    written = []

    path = _prefixed(figures_dir, label, "paced_latency_distribution.png")
    plot_latency_distribution(fp32, int8, path)
    written.append(path)

    if include_latency_over_record:
        path = _prefixed(figures_dir, label, "paced_latency_over_record.png")
        _stacked_panels(
            _latency_ms(fp32),
            _latency_ms(int8),
            y_label="Processing latency (ms)",
            title="Chunk processing latency across the paced record",
            output_path=path,
            reference_ms=chunk_period_ms,
        )
        written.append(path)

    path = _prefixed(figures_dir, label, "paced_scheduling_lateness.png")
    _stacked_panels(
        _lateness_ms(fp32),
        _lateness_ms(int8),
        y_label="Scheduling lateness (ms)",
        title="Scheduling lateness across the paced record",
        output_path=path,
    )
    written.append(path)

    return written


# ---------------------------------------------------------------------
#                    Governor Comparison Figure
# ---------------------------------------------------------------------


def write_governor_comparison(
    baseline_fp32_npz: Path,
    baseline_int8_npz: Path,
    current_fp32_npz: Path,
    current_int8_npz: Path,
    chunk_period_ms: float,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    baseline_label: str = "ondemand",
    current_label: str = "performance",
) -> Path:
    """
    2x2 latency-over-record grid: rows are models, columns are governor
    configurations, with deadline misses recomputed from raw timestamps
    and annotated per panel. All panels share axes, so the baseline's
    excursions over the chunk period are directly comparable with the
    fixed configuration's headroom.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        ("FP32", baseline_label): _load_paced(baseline_fp32_npz),
        ("INT8", baseline_label): _load_paced(baseline_int8_npz),
        ("FP32", current_label): _load_paced(current_fp32_npz),
        ("INT8", current_label): _load_paced(current_int8_npz),
    }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 7),
        sharex=True,
        sharey=True,
    )
    colours = {"FP32": FP32_COLOUR, "INT8": INT8_COLOUR}

    for row, model in enumerate(("FP32", "INT8")):
        for column, configuration in enumerate((baseline_label, current_label)):
            axis = axes[row][column]
            arrays = runs[(model, configuration)]
            misses = _deadline_misses(arrays, chunk_period_ms)

            axis.plot(
                _latency_ms(arrays),
                linewidth=0.5,
                color=colours[model],
            )
            axis.axhline(
                chunk_period_ms,
                color="red",
                linestyle="--",
                linewidth=1,
            )
            axis.set_title(
                f"{model} — {configuration} "
                f"({misses} deadline miss{'es' if misses != 1 else ''})",
                fontsize=10,
            )

            if column == 0:
                axis.set_ylabel("Processing latency (ms)")

            if row == 1:
                axis.set_xlabel("Chunk index")

    figure.suptitle(
        "Paced chunk latency by CPU governor "
        f"({baseline_label} vs {current_label}); dashed line = chunk period"
    )
    figure.tight_layout()

    output_path = figures_dir / "governor_comparison_latency.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Plot Section 5.3 paced-run artifacts on the dev machine."
    )
    parser.add_argument("--fp32-npz", type=Path, required=True)
    parser.add_argument("--int8-npz", type=Path, required=True)
    parser.add_argument("--chunk-period-ms", type=float, default=100.0)
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Filename prefix for this configuration (e.g. performance).",
    )
    parser.add_argument(
        "--baseline-fp32-npz",
        type=Path,
        default=None,
        help="Baseline-configuration FP32 NPZ for the governor comparison.",
    )
    parser.add_argument(
        "--baseline-int8-npz",
        type=Path,
        default=None,
        help="Baseline-configuration INT8 NPZ for the governor comparison.",
    )
    parser.add_argument("--baseline-label", type=str, default="ondemand")
    parser.add_argument(
        "--no-latency-over-record",
        action="store_true",
        help=(
            "Skip the per-configuration latency-over-record figure, for "
            "runs whose panels the governor comparison already shows."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )

    args = parser.parse_args()

    has_baseline = (
        args.baseline_fp32_npz is not None and args.baseline_int8_npz is not None
    )

    written = write_realtime_figures(
        fp32_npz_path=args.fp32_npz,
        int8_npz_path=args.int8_npz,
        chunk_period_ms=args.chunk_period_ms,
        figures_dir=args.figures_dir,
        label=args.label,
        # The governor comparison already shows every over-record panel.
        include_latency_over_record=(
            not has_baseline and not args.no_latency_over_record
        ),
    )

    if has_baseline:
        written.append(
            write_governor_comparison(
                baseline_fp32_npz=args.baseline_fp32_npz,
                baseline_int8_npz=args.baseline_int8_npz,
                current_fp32_npz=args.fp32_npz,
                current_int8_npz=args.int8_npz,
                chunk_period_ms=args.chunk_period_ms,
                figures_dir=args.figures_dir,
                baseline_label=args.baseline_label,
                current_label=args.label or "performance",
            )
        )

    for path in written:
        logger.info("Wrote figure %s", path)


if __name__ == "__main__":
    main()
