"""Optional plots explaining where two detection timelines disagree."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.preprocessing.beat_extraction import SAMPLING_RATE

# Seconds of ECG drawn either side of the first divergent detection.
CONTEXT_SECONDS = 5.0


def plot_peak_divergence(
    record_name: str,
    signal: NDArray[np.float64],
    whole_record_peaks: NDArray[np.int64],
    streaming_peaks: NDArray[np.int64],
    divergent_peaks: list[int],
    output_dir: Path,
) -> Path:
    """
    Plot the ECG around the first divergent detection for one record.

    Called only when the whole-record and streaming timelines actually
    disagree, so no plot is produced for a record at exact peak parity.
    """

    if not divergent_peaks:
        raise ValueError(f"Record {record_name} has no divergent peaks to plot.")

    context = int(CONTEXT_SECONDS * SAMPLING_RATE)
    centre = min(divergent_peaks)
    start = max(0, centre - context)
    stop = min(len(signal), centre + context)
    positions = np.arange(start, stop)

    def visible(peaks: NDArray[np.int64]) -> NDArray[np.int64]:
        peaks = np.asarray(peaks, dtype=np.int64)
        return peaks[(peaks >= start) & (peaks < stop)]

    plt.figure(figsize=(14, 5))
    plt.plot(positions, signal[start:stop], label="ECG signal", linewidth=1.0)

    whole_visible = visible(whole_record_peaks)
    stream_visible = visible(streaming_peaks)

    plt.scatter(
        whole_visible,
        signal[whole_visible],
        marker="x",
        s=90,
        color="green",
        label="Whole-record XQRS",
        zorder=5,
    )
    plt.scatter(
        stream_visible,
        signal[stream_visible],
        marker="o",
        facecolors="none",
        edgecolors="red",
        s=130,
        label="Streaming XQRS",
        zorder=4,
    )

    for peak in divergent_peaks:
        if start <= peak < stop:
            plt.axvline(peak, color="orange", alpha=0.5, linestyle="--")

    plt.title(f"Record {record_name}: causal versus whole-record detection")
    plt.xlabel("Absolute sample index")
    plt.ylabel("ECG amplitude")
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"record_{record_name}_peak_divergence.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
