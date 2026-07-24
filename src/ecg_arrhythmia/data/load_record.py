import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from ecg_arrhythmia.preprocessing.beat_extraction import BEAT_SYMBOLS

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def load_record(
    record_name: str,
) -> tuple[np.ndarray, dict[str, Any], wfdb.Annotation]:
    """
    Load the ECG signals, metadata, and expert annotations for one
    record from the MIT-BIH Arrhythmia Database.
    """

    # Load both ECG channels for the requested record.
    #
    # signals is a 2D NumPy array with shape:
    # (number_of_samples, number_of_channels)
    #
    # Each row represents one moment in time, each column represents
    # a different way of measuring heart rate activity (ECG lead).
    # MIT-BIH records contain two channels.
    #
    # fields contains metadata such as the sampling frequency and lead names.
    signals, fields = wfdb.rdsamp(
        record_name=record_name,
        pn_dir="mitdb",
    )

    # Load the expert annotations associated with the record.
    # annotation.sample contains the ECG sample positions of annotations.
    # annotation.symbol contains the corresponding annotation symbols,
    # such as "N" for a normal beat or "V" for a premature ventricular beat.
    annotation = wfdb.rdann(
        record_name=record_name,
        extension="atr",
        pn_dir="mitdb",
    )

    # Signals can be both a numpy array or None.
    if signals is None:
        raise ValueError("No signal found")

    # MIT-BIH should return two-dimensional signal data.
    if signals.ndim != 2:
        raise ValueError(f"Expected a 2D signal array, received shape {signals.shape}")

    # Annotation.symbol is allowed to be None by the WFDB Annotation class.
    # We need the symbols because they will become our labels later on.
    if annotation.symbol is None:
        raise ValueError(f"Record {record_name} contains no annotation symbols")

    # Log useful information for checking that the record loaded correctly.
    logger.debug("Record: %s", record_name)
    logger.debug("Signal shape: %s", signals.shape)
    logger.debug("Sampling frequency: %s", fields["fs"])
    logger.debug("Signal names: %s", fields["sig_name"])
    logger.debug("First annotation samples: %s", annotation.sample[:10])
    logger.debug("First annotation symbols: %s", annotation.symbol[:10])

    return signals, fields, annotation


def select_signal_channel(
    signals: np.ndarray,
    fields: dict[str, Any],
    preferred_lead: str = "MLII",
) -> tuple[np.ndarray, str]:
    """
    Select one ECG channel from the two channels stored in a record.

    MLII is preferred because it is available in most MIT-BIH records.
    If it is unavailable, the first channel is used instead.
    """

    # The lead names are stored in the same order as the signal columns.
    # For example if fields["sig_name"] == ["MLII", "V5"] then
    # signals[:, 0] is therefore MLII and signals[:, 1] is V5.
    signal_names = fields["sig_name"]

    # Find the column containing the preferred lead.
    if preferred_lead in signal_names:
        channel_index = signal_names.index(preferred_lead)

    # Some records, such as 102 and 104, do not contain MLII.
    # In that case, just use the first available channel.
    else:
        channel_index = 0

        logger.warning(
            "%s is unavailable; using %s instead",
            preferred_lead,
            signal_names[channel_index],
        )

    # Extract every ECG amplitude measurement from the selected channel.
    # This converts the 2D signal matrix into a 1D signal array.
    signal = signals[:, channel_index]

    # Store the selected lead name so we know how the signal was measured.
    lead_name = signal_names[channel_index]

    return signal, lead_name


def plot_record(
    record_name: str, signal: np.ndarray, annotation: wfdb.Annotation, lead_name: str
) -> None:

    # Plot the first 3000 amplitudes/ 3000/360 ≈ 8.3 seconds
    # of ECG recording for this record
    start = 0
    end = 3000

    if annotation.symbol is None:
        raise ValueError("No annotation for this sample.")

    plt.figure(figsize=(12, 4))
    plt.plot(signal[start:end], label="ECG signal")

    # Gives (sample_index, symbol at that index)
    for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True):
        # The sample index has to be between the start and end index
        if start <= sample < end:
            # Draw a vertical red line at x position sample - start
            plt.axvline(sample - start, color="red", alpha=0.3)
            # Put symbol at x postion sample - start, and y position signal[sample].
            plt.text(
                sample - start, float(signal[sample]), symbol, color="green", fontsize=8
            )

    plt.title(f"MIT-BIH Record {record_name}: {lead_name} Signal with Beat Annotations")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ecg_plot.png")
    plt.close()


def plot_r_peak_comparison(
    record_name: str,
    signal: np.ndarray,
    annotation: wfdb.Annotation,
    detected_peaks: np.ndarray,
    lead_name: str,
    detector_name: str,
    start: int = 0,
    end: int = 3000,
    output_path: str | Path = "r_peak_comparison.png",
) -> None:
    """
    Plot an ECG segment with expert heartbeat annotations and detected
    R-peak positions.

    Parameters
    ----------
    record_name:
        Name of the MIT-BIH record being plotted.

    signal:
        One-dimensional ECG signal for the selected lead.

    annotation:
        Expert WFDB annotations for the record.

    detected_peaks:
        Absolute sample indices returned by an R-peak detector.

    lead_name:
        Name of the selected ECG lead.

    detector_name:
        Name of the R-peak detector used.

    start:
        First absolute sample index included in the plot.

    end:
        First absolute sample index excluded from the plot.

    output_path:
        Location where the generated plot should be saved.
    """

    if signal.ndim != 1:
        raise ValueError(
            f"ECG signal must be one-dimensional, but received shape {signal.shape}."
        )

    if annotation.symbol is None:
        raise ValueError(f"Record {record_name} contains no annotation symbols.")

    if start < 0:
        raise ValueError("Plot start index must not be negative.")

    if end <= start:
        raise ValueError("Plot end index must be greater than the start index.")

    if end > len(signal):
        raise ValueError(f"Plot end index {end} exceeds signal length {len(signal)}.")

    detected_peaks = np.asarray(detected_peaks, dtype=np.int64)

    # Keep only expert annotations representing genuine heartbeats and
    # lying inside the requested plot range.
    visible_expert_beats = [
        (sample, symbol)
        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol,
            strict=True,
        )
        if symbol in BEAT_SYMBOLS and start <= sample < end
    ]

    expert_samples = np.array(
        [sample for sample, _ in visible_expert_beats],
        dtype=np.int64,
    )

    expert_symbols = [symbol for _, symbol in visible_expert_beats]

    # Keep only detector outputs lying inside the requested plot range.
    visible_detected_peaks = detected_peaks[
        (detected_peaks >= start) & (detected_peaks < end)
    ]

    # Use absolute sample positions on the x-axis.
    sample_positions = np.arange(start, end)

    plt.figure(figsize=(14, 5))

    plt.plot(
        sample_positions,
        signal[start:end],
        label=f"{lead_name} ECG signal",
    )

    if expert_samples.size > 0:
        # Mark the expert annotation positions directly on the ECG.
        plt.scatter(
            expert_samples,
            signal[expert_samples],
            marker="x",
            s=80,
            color="green",
            linewidths=2,
            label="Expert annotations",
            zorder=5,
        )

        # Display the expert beat symbol beside each annotation.
        for sample, symbol in zip(
            expert_samples,
            expert_symbols,
            strict=True,
        ):
            plt.text(
                sample,
                float(signal[sample]),
                f" {symbol}",
                fontsize=8,
                verticalalignment="bottom",
            )

    if visible_detected_peaks.size > 0:
        # Use hollow circular markers so detections can be compared
        # directly with the expert annotation markers.
        plt.scatter(
            visible_detected_peaks,
            signal[visible_detected_peaks],
            marker="o",
            facecolors="none",
            edgecolors="red",
            s=120,
            linewidths=2,
            label=f"{detector_name.upper()} detections",
            zorder=4,
        )

    plt.title(
        f"MIT-BIH Record {record_name}: {detector_name.upper()} R-Peak Comparison"
    )
    plt.xlabel("Absolute sample index")
    plt.ylabel("ECG amplitude")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
