import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

FP32_COLOUR = "steelblue"
INT8_COLOUR = "darkorange"
BOUNDARY_COLOUR = "0.35"

# Slightly larger than the panel y-labels (9 pt) so the record spans
# stay legible at dissertation print size.
RECORD_LABEL_FONTSIZE = 10

DEFAULT_FIGURES_DIR = Path("artifacts/figures/edge_sustained_resources")

SECONDS_PER_MINUTE = 60.0


def _load(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {name: data[name] for name in data.files}


def _minutes(telemetry: dict[str, np.ndarray]) -> np.ndarray:
    return telemetry["elapsed_seconds"] / 60.0


# ---------------------------------------------------------------------
#                        Record Boundaries
# ---------------------------------------------------------------------


def record_segments(per_record: list[dict]) -> list[dict]:
    """
    Cumulative start, end and midpoint minutes for each record replayed.

    Durations come from each record's `record_wall_seconds`, the paced
    duration measured during the run. Paced replay advances one chunk
    period per chunk, so this equals the record's ECG signal time to
    within one chunk period, and it shares the telemetry's clock — the
    result JSON does not store a sampling rate, so samples divided by
    rate is not directly available.
    """

    segments = []
    elapsed_seconds = 0.0

    for record in per_record:
        start_seconds = elapsed_seconds
        elapsed_seconds += float(record["record_wall_seconds"])

        segments.append(
            {
                "record_name": record["record_name"],
                "start_minutes": start_seconds / SECONDS_PER_MINUTE,
                "end_minutes": elapsed_seconds / SECONDS_PER_MINUTE,
                "midpoint_minutes": (start_seconds + elapsed_seconds)
                / 2.0
                / SECONDS_PER_MINUTE,
            }
        )

    return segments


def transition_minutes(segments: list[dict]) -> list[float]:
    """
    Interior record transitions only.

    The start of the run and the end of the final (possibly truncated)
    segment are not transitions, so neither is emitted.
    """

    return [segment["end_minutes"] for segment in segments[:-1]]


def load_record_segments(json_path: Path) -> list[dict] | None:
    """Read per-record spans from a sustained-run result JSON."""

    try:
        with json_path.open("r", encoding="utf-8") as file:
            per_record = json.load(file)["streaming"]["per_record"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        logger.warning("No usable per-record metadata in %s", json_path)
        return None

    if not per_record:
        return None

    return record_segments(per_record)


def default_result_json(npz_path: Path) -> Path | None:
    """The result JSON written beside a `<stem>_raw.npz` telemetry file."""

    suffix = "_raw.npz"

    if not npz_path.name.endswith(suffix):
        return None

    candidate = npz_path.with_name(npz_path.name[: -len(suffix)] + ".json")

    return candidate if candidate.exists() else None


def _draw_record_boundaries(figure, axes, segments: list[dict]) -> None:
    """
    Dashed transitions in every panel, record names on a top axis.

    Names sit at each span's midpoint rather than on the lines, so it
    is unambiguous which interval belongs to which record, and appear
    once for the figure rather than in every panel.
    """

    for axis in axes:
        for boundary in transition_minutes(segments):
            axis.axvline(
                boundary,
                color=BOUNDARY_COLOUR,
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                zorder=0.5,
            )

    label_axis = axes[0].twiny()
    label_axis.set_xlim(axes[0].get_xlim())
    label_axis.set_xticks([segment["midpoint_minutes"] for segment in segments])
    label_axis.set_xticklabels(
        [segment["record_name"] for segment in segments],
        fontsize=RECORD_LABEL_FONTSIZE,
    )
    label_axis.tick_params(axis="x", length=0)


def write_precision_timeseries(
    npz_path: Path,
    precision: str,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    result_json_path: Path | None = None,
) -> Path:
    """
    Aligned temperature/frequency/RSS/CPU panels for one precision.

    Record boundaries are drawn when the run's result JSON is supplied
    or found beside the NPZ; without it the figure is unchanged.
    """

    figures_dir.mkdir(parents=True, exist_ok=True)

    telemetry = _load(npz_path)
    minutes = _minutes(telemetry)
    colour = FP32_COLOUR if precision.lower() == "fp32" else INT8_COLOUR

    if result_json_path is None:
        result_json_path = default_result_json(npz_path)

    segments = (
        load_record_segments(result_json_path) if result_json_path is not None else None
    )

    panels = (
        ("temperature_c", "Temperature (°C)", 1.0),
        ("cpu_frequency_khz", "CPU frequency (MHz)", 1 / 1000.0),
        ("rss_mib", "Process RSS (MiB)", 1.0),
        ("process_cpu_percent", "Process CPU (% of one core)", 1.0),
    )

    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10, 9),
        sharex=True,
    )

    for axis, (key, label, scale) in zip(axes, panels, strict=True):
        axis.plot(minutes, telemetry[key] * scale, color=colour, linewidth=1.0)
        axis.set_ylabel(label, fontsize=9)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Elapsed time (minutes)")

    if segments:
        _draw_record_boundaries(figure, axes, segments)

    figure.suptitle(
        f"Sustained {precision.upper()} run: resource telemetry over "
        f"{minutes[-1]:.0f} minutes"
    )
    figure.tight_layout()

    output_path = figures_dir / f"{precision.lower()}_sustained_timeseries.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path


def write_precision_comparison(
    fp32_npz_path: Path,
    int8_npz_path: Path,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> Path:
    """Temperature and RSS side by side for both precisions."""

    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "FP32": (_load(fp32_npz_path), FP32_COLOUR),
        "INT8": (_load(int8_npz_path), INT8_COLOUR),
    }

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for label, (telemetry, colour) in runs.items():
        minutes = _minutes(telemetry)
        axes[0].plot(
            minutes,
            telemetry["temperature_c"],
            color=colour,
            linewidth=1.0,
            label=label,
        )
        axes[1].plot(
            minutes,
            telemetry["rss_mib"],
            color=colour,
            linewidth=1.0,
            label=label,
        )

    axes[0].set_ylabel("Temperature (°C)")
    axes[1].set_ylabel("Process RSS (MiB)")
    axes[1].set_xlabel("Elapsed time (minutes)")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("Sustained runs: FP32 vs INT8 temperature and memory")
    figure.tight_layout()

    output_path = figures_dir / "sustained_fp32_vs_int8_comparison.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Plot Section 5.4 sustained telemetry on the dev machine."
    )
    parser.add_argument("--fp32-npz", type=Path, required=True)
    parser.add_argument("--int8-npz", type=Path, required=True)
    parser.add_argument(
        "--fp32-json",
        type=Path,
        default=None,
        help="Result JSON for record boundaries (found beside the NPZ by default).",
    )
    parser.add_argument(
        "--int8-json",
        type=Path,
        default=None,
        help="Result JSON for record boundaries (found beside the NPZ by default).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
    )

    args = parser.parse_args()

    written = [
        write_precision_timeseries(
            args.fp32_npz,
            "fp32",
            args.figures_dir,
            args.fp32_json,
        ),
        write_precision_timeseries(
            args.int8_npz,
            "int8",
            args.figures_dir,
            args.int8_json,
        ),
        write_precision_comparison(
            args.fp32_npz,
            args.int8_npz,
            args.figures_dir,
        ),
    ]

    for path in written:
        logger.info("Wrote figure %s", path)


if __name__ == "__main__":
    main()
