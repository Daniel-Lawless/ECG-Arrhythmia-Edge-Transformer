import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.data.build_dataset import EXCLUDED_AAMI_LABELS
from ecg_arrhythmia.data.build_xqrs_centered_dataset import (
    MATCHING_TOLERANCE_MS,
    build_record_detected_beats,
    load_split_record_names,
)
from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.data.sequence_dataset import create_record_sequences
from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.detection.xqrs_detector import XQRSDetector
from ecg_arrhythmia.evaluation.r_peak_matching import match_r_peaks
from ecg_arrhythmia.preprocessing.beat_extraction import (
    LOCAL_RR_WINDOW,
    SEQUENCE_LENGTH,
)
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplaySource
from ecg_arrhythmia.streaming.streaming_engine import (
    StreamContinuityError,
    StreamingEngine,
)

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/results/streaming_evaluation")
DIAGNOSTICS_DIR = Path("artifacts/figures/streaming_diagnostics")

# Tolerance used only to pair the two detection timelines for reporting.
PEAK_TOLERANCE_MS = 100.0


# ---------------------------------------------------------------------
#                       Offline And Streaming Views
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Sequence:
    """One sequence reduced to the parts parity actually compares."""

    ecg: NDArray[np.float64]  # (sequence_length, window_size)
    rr: NDArray[np.float64]  # (sequence_length, 2)
    peaks: tuple[int, ...]


@dataclass(frozen=True)
class _OfflineReference:
    """Every sequence the offline builder can form for one record."""

    beat_peaks: NDArray[np.int64]
    beat_index_of_peak: dict[int, int]
    sequences: dict[int, _Sequence]
    scored_peaks: set[int]


@dataclass(frozen=True)
class _StreamingRun:
    """Outcome of replaying one record through the streaming engine."""

    sequences: dict[int, _Sequence]
    peaks: NDArray[np.int64]
    total_input_samples: int
    samples_accepted: int
    continuity_validated: bool


class _ReplayedDetector(RPeakDetector):
    """Serves a precomputed timeline so XQRS runs once per record."""

    def __init__(self, peaks: NDArray[np.int64]) -> None:
        self._peaks = np.asarray(peaks, dtype=np.int64)
        self.learn = True

    @property
    def name(self) -> str:
        return "xqrs"

    def _detect(
        self,
        signal: NDArray[np.float64],
        sampling_rate: float,
    ) -> NDArray[np.int64]:
        return self._peaks


def _offline_reference(
    record_name: str,
    whole_record_peaks: NDArray[np.int64],
) -> _OfflineReference:
    """Rebuild the offline XQRS-centred sequences for one record."""

    conversion = build_record_detected_beats(
        record_name=record_name,
        detector=_ReplayedDetector(whole_record_peaks),
        tolerance_ms=MATCHING_TOLERANCE_MS,
        normalise_beats=False,
        excluded_labels=set(EXCLUDED_AAMI_LABELS),
    )

    num_beats = conversion.num_valid_beats
    record_metadata = [
        {
            "record_id": conversion.record_id,
            "patient_id": conversion.patient_id,
            "start_index": 0,
            "end_index": num_beats,
            "num_beats": num_beats,
        }
    ]

    ecg_sequences, _, rr_sequences, _, target_indices, _ = create_record_sequences(
        X=conversion.windows,
        y=conversion.labels_for_sequencing,
        rr_features=conversion.rr_features,
        patient_ids=np.asarray([conversion.patient_id] * num_beats, dtype=str),
        record_metadata=record_metadata,
        sequence_length=SEQUENCE_LENGTH,
    )

    beat_peaks = conversion.detected_samples.astype(np.int64)
    targets = [int(index) for index in target_indices]

    sequences = {
        int(beat_peaks[target]): _Sequence(
            ecg=ecg_sequences[row],
            rr=rr_sequences[row],
            peaks=tuple(
                int(peak)
                for peak in beat_peaks[target - SEQUENCE_LENGTH + 1 : target + 1]
            ),
        )
        for row, target in enumerate(targets)
    }

    return _OfflineReference(
        beat_peaks=beat_peaks,
        beat_index_of_peak={int(peak): i for i, peak in enumerate(beat_peaks)},
        sequences=sequences,
        scored_peaks={
            int(beat_peaks[target])
            for target in targets
            if bool(conversion.is_target[target])
        },
    )


def _replay(
    signal: NDArray[np.float64],
    sampling_rate: float,
    record_name: str,
    chunk_size: int,
) -> _StreamingRun:
    """Push one record through the streaming engine and collect its output."""

    source = ReplaySource(
        signal=signal,
        sampling_rate=sampling_rate,
        chunk_size=chunk_size,
        record_name=record_name,
    )
    engine = StreamingEngine()
    engine.start_record(record_name=record_name)

    sequences: dict[int, _Sequence] = {}
    peaks: list[int] = []
    continuity_validated = True

    def collect(emitted: list) -> None:
        for sequence in emitted:
            sequences[int(sequence.target_peak_index)] = _Sequence(
                ecg=np.squeeze(sequence.ecg, axis=1),
                rr=sequence.rr,
                peaks=tuple(int(peak) for peak in sequence.peak_indices),
            )
        peaks.extend(engine.last_confirmed_peaks)

    try:
        for chunk in source.iter_chunks():
            collect(engine.process_chunk(chunk))
        collect(engine.flush())
    except StreamContinuityError:
        continuity_validated = False
        logger.exception("Stream continuity failed for record %s", record_name)

    return _StreamingRun(
        sequences=sequences,
        peaks=np.asarray(peaks, dtype=np.int64),
        total_input_samples=source.num_samples,
        samples_accepted=engine.state.total_samples_accepted,
        continuity_validated=continuity_validated,
    )


# ---------------------------------------------------------------------
#                              Comparison
# ---------------------------------------------------------------------


def _compare_peaks(
    whole_record_peaks: NDArray[np.int64],
    streaming_peaks: NDArray[np.int64],
    sampling_rate: float,
) -> dict:
    """Pair the whole-record and streaming detection timelines."""

    tolerance = round(PEAK_TOLERANCE_MS * sampling_rate / 1000.0)
    match = match_r_peaks(
        annotation_samples=whole_record_peaks,
        detected_samples=streaming_peaks,
        tolerance_samples=tolerance,
    )
    offsets = match.offsets_samples

    return {
        "whole_record_peaks": int(whole_record_peaks.size),
        "streaming_peaks": int(streaming_peaks.size),
        "exact_peak_matches": int(np.sum(offsets == 0)) if offsets.size else 0,
        "largest_absolute_peak_offset": (
            int(np.max(np.abs(offsets))) if offsets.size else 0
        ),
        "missing_peaks": [
            int(peak) for peak in whole_record_peaks[match.unmatched_annotation_indices]
        ],
        "extra_peaks": [
            int(peak) for peak in streaming_peaks[match.unmatched_detection_indices]
        ],
    }


def _rr_history_diverged(
    offline: _OfflineReference,
    target_peak: int,
    whole_record_peaks: NDArray[np.int64],
    streaming_peaks: NDArray[np.int64],
) -> bool:
    """
    Do the two detection timelines differ across this target's RR history?

    An RR feature depends on the preceding detection and on the last
    LOCAL_RR_WINDOW completed beats, so the dependency reaches further back
    than the five beats the model sees. The lookback therefore spans the
    whole sequence plus that local window, with one extra detection for the
    interval feeding the oldest beat considered.
    """

    target_index = offline.beat_index_of_peak[target_peak]
    lookback = SEQUENCE_LENGTH - 1 + LOCAL_RR_WINDOW + 1
    history_start = int(offline.beat_peaks[max(0, target_index - lookback)])

    def within(peaks: NDArray[np.int64]) -> NDArray[np.int64]:
        return peaks[(peaks >= history_start) & (peaks <= target_peak)]

    return not np.array_equal(within(whole_record_peaks), within(streaming_peaks))


def _compare_sequences(
    offline: _OfflineReference,
    run: _StreamingRun,
    whole_record_peaks: NDArray[np.int64],
) -> dict:
    """Compare emitted sequences with the offline targets and explain gaps."""

    emitted = set(run.sequences)
    whole_peak_set = {int(peak) for peak in whole_record_peaks}
    streaming_peak_set = {int(peak) for peak in run.peaks}

    peak_index_mismatches: list[int] = []
    ecg_mismatches: list[int] = []
    rr_mismatches: list[int] = []
    unexplained_content: set[int] = set()
    exactly_matched = 0
    ecg_explained = 0
    rr_explained = 0

    for target_peak in sorted(offline.scored_peaks & emitted):
        expected = offline.sequences[target_peak]
        actual = run.sequences[target_peak]

        same_peaks = expected.peaks == actual.peaks
        same_ecg = np.array_equal(expected.ecg, actual.ecg)
        same_rr = np.array_equal(expected.rr, actual.rr)

        if same_ecg and same_rr:
            exactly_matched += 1

        if not same_peaks:
            peak_index_mismatches.append(target_peak)

        if not same_ecg:
            ecg_mismatches.append(target_peak)
            # Only a changed set of beats can change the ECG windows.
            if same_peaks:
                unexplained_content.add(target_peak)
            else:
                ecg_explained += 1

        if not same_rr:
            rr_mismatches.append(target_peak)
            diverged = not same_peaks or _rr_history_diverged(
                offline=offline,
                target_peak=target_peak,
                whole_record_peaks=whole_record_peaks,
                streaming_peaks=run.peaks,
            )
            if diverged:
                rr_explained += 1
            else:
                unexplained_content.add(target_peak)

    deployment_only: list[int] = []
    causal_only: list[int] = []
    unexplained_extra: list[int] = []

    for target_peak in sorted(emitted - offline.scored_peaks):
        if target_peak in offline.sequences:
            # A valid offline sequence exists; only the expert-label filter
            # kept it out of the scored dataset.
            deployment_only.append(target_peak)
        elif target_peak not in whole_peak_set:
            causal_only.append(target_peak)
        else:
            unexplained_extra.append(target_peak)

    missing_causal: list[int] = []
    unexplained_missing: list[int] = []

    for target_peak in sorted(offline.scored_peaks - emitted):
        if target_peak not in streaming_peak_set:
            missing_causal.append(target_peak)
        else:
            unexplained_missing.append(target_peak)

    return {
        "offline_targets_expected": len(offline.scored_peaks),
        "streaming_sequences_emitted": len(run.sequences),
        "exactly_matched_sequences": exactly_matched,
        "missing_targets": sorted(offline.scored_peaks - emitted),
        "extra_targets": sorted(emitted - offline.scored_peaks),
        "peak_index_mismatches": peak_index_mismatches,
        "ecg_window_mismatches": ecg_mismatches,
        "rr_feature_mismatches": rr_mismatches,
        "expected_deployment_only_targets": deployment_only,
        "causal_detector_only_targets": causal_only,
        "missing_targets_due_to_causal_divergence": missing_causal,
        "unexplained_extra_targets": unexplained_extra,
        "unexplained_missing_targets": unexplained_missing,
        "ecg_mismatches_explained_by_peak_history": ecg_explained,
        "rr_mismatches_explained_by_peak_history": rr_explained,
        "unexplained_content_mismatches": sorted(unexplained_content),
    }


# ---------------------------------------------------------------------
#                            Per-Record Entry
# ---------------------------------------------------------------------


def evaluate_record(
    record_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    write_diagnostics: bool = False,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> dict:
    """Evaluate streaming parity for one record and return its summary."""

    signals, fields, _ = load_record(record_name=record_name)
    signal, _ = select_signal_channel(signals=signals, fields=fields)
    sampling_rate = float(fields["fs"])

    whole_record_peaks = XQRSDetector(learn=True).detect(
        signal=signal,
        sampling_rate=sampling_rate,
    )
    offline = _offline_reference(record_name, whole_record_peaks)
    run = _replay(
        signal=signal,
        sampling_rate=sampling_rate,
        record_name=record_name,
        chunk_size=chunk_size,
    )

    peaks = _compare_peaks(whole_record_peaks, run.peaks, sampling_rate)
    sequences = _compare_sequences(offline, run, whole_record_peaks)

    result = {
        "record_name": record_name,
        "sampling_rate": sampling_rate,
        "chunk_size": chunk_size,
        "total_input_samples": run.total_input_samples,
        "samples_accepted": run.samples_accepted,
        "continuity_validated": run.continuity_validated,
        **peaks,
        **sequences,
    }
    result["exact_parity"] = _is_exact_parity(result)
    result["all_differences_explained"] = _is_explained(result)

    if write_diagnostics and (peaks["missing_peaks"] or peaks["extra_peaks"]):
        from ecg_arrhythmia.evaluation.streaming_diagnostics import (
            plot_peak_divergence,
        )

        result["diagnostics_plot"] = str(
            plot_peak_divergence(
                record_name=record_name,
                signal=signal,
                whole_record_peaks=whole_record_peaks,
                streaming_peaks=run.peaks,
                divergent_peaks=peaks["missing_peaks"] + peaks["extra_peaks"],
                output_dir=diagnostics_dir,
            )
        )

    return result


def _is_exact_parity(result: dict) -> bool:
    """
    True when streaming reproduced the offline record with no differences.

    A record whose stream broke can never be exact, however few
    differences were collected before the failure.
    """

    if not result["continuity_validated"]:
        return False

    return not any(
        result[field]
        for field in (
            "missing_peaks",
            "extra_peaks",
            "missing_targets",
            "extra_targets",
            "peak_index_mismatches",
            "ecg_window_mismatches",
            "rr_feature_mismatches",
        )
    )


def _is_explained(result: dict) -> bool:
    """
    True when every difference has a known cause.

    A broken stream is itself an unexplained difference, because the
    comparison never saw the whole record.
    """

    if not result["continuity_validated"]:
        return False

    return not any(
        result[field]
        for field in (
            "unexplained_extra_targets",
            "unexplained_missing_targets",
            "unexplained_content_mismatches",
        )
    )


# ---------------------------------------------------------------------
#                              Aggregate
# ---------------------------------------------------------------------

# Aggregate fields summed straight from the per-record scalars.
_SCALAR_TOTALS = {
    "total_input_samples": "total_input_samples",
    "total_samples_accepted": "samples_accepted",
    "total_whole_record_peaks": "whole_record_peaks",
    "total_streaming_peaks": "streaming_peaks",
    "total_exact_peak_matches": "exact_peak_matches",
    "total_offline_targets_expected": "offline_targets_expected",
    "total_streaming_sequences_emitted": "streaming_sequences_emitted",
    "total_exactly_matched_sequences": "exactly_matched_sequences",
    "total_ecg_mismatches_explained_by_peak_history": (
        "ecg_mismatches_explained_by_peak_history"
    ),
    "total_rr_mismatches_explained_by_peak_history": (
        "rr_mismatches_explained_by_peak_history"
    ),
}

# Aggregate fields counted from the per-record diagnostic lists.
_LIST_TOTALS = {
    "total_missing_peaks": "missing_peaks",
    "total_extra_peaks": "extra_peaks",
    "total_missing_targets": "missing_targets",
    "total_extra_targets": "extra_targets",
    "total_peak_index_mismatches": "peak_index_mismatches",
    "total_ecg_window_mismatches": "ecg_window_mismatches",
    "total_rr_feature_mismatches": "rr_feature_mismatches",
    "total_expected_deployment_only_targets": "expected_deployment_only_targets",
    "total_causal_detector_only_targets": "causal_detector_only_targets",
    "total_missing_targets_due_to_causal_divergence": (
        "missing_targets_due_to_causal_divergence"
    ),
    "total_unexplained_extra_targets": "unexplained_extra_targets",
    "total_unexplained_missing_targets": "unexplained_missing_targets",
    "total_unexplained_content_mismatches": "unexplained_content_mismatches",
}


def aggregate_records(
    record_results: list[dict],
    failed_records: list[str] | None = None,
) -> dict:
    """
    Combine per-record summaries into raw and interpreted parity totals.

    Pure over the per-record dictionaries, so the interpretation can be
    tested with synthetic summaries and never has to replay a record.
    """

    failed_records = list(failed_records or [])

    aggregate: dict = {
        "num_records_evaluated": len(record_results),
        "record_names": [result["record_name"] for result in record_results],
        "chunk_size": record_results[0]["chunk_size"] if record_results else None,
        "sampling_rates": {
            result["record_name"]: result["sampling_rate"] for result in record_results
        },
    }

    for name, field in _SCALAR_TOTALS.items():
        aggregate[name] = sum(int(result[field]) for result in record_results)

    for name, field in _LIST_TOTALS.items():
        aggregate[name] = sum(len(result[field]) for result in record_results)

    aggregate["largest_absolute_peak_offset"] = max(
        (int(result["largest_absolute_peak_offset"]) for result in record_results),
        default=0,
    )
    aggregate["records_with_perfect_peak_parity"] = [
        result["record_name"]
        for result in record_results
        if not result["missing_peaks"] and not result["extra_peaks"]
    ]
    aggregate["records_with_perfect_sequence_parity"] = [
        result["record_name"] for result in record_results if result["exact_parity"]
    ]
    aggregate["failed_records"] = failed_records
    aggregate["all_records_continuity_validated"] = all(
        result["continuity_validated"] for result in record_results
    )

    # No verdict is meaningful unless every requested record was actually
    # evaluated and arrived as one unbroken stream.
    evaluated_cleanly = (
        bool(record_results)
        and not failed_records
        and aggregate["all_records_continuity_validated"]
    )

    aggregate["all_records_exact_parity"] = evaluated_cleanly and all(
        result["exact_parity"] for result in record_results
    )

    explained = [
        result["record_name"]
        for result in record_results
        if result["all_differences_explained"]
    ]
    aggregate["num_records_with_all_differences_explained"] = len(explained)
    aggregate["records_with_unexplained_differences"] = [
        result["record_name"]
        for result in record_results
        if not result["all_differences_explained"]
    ]
    aggregate["all_records_differences_explained"] = (
        evaluated_cleanly and not aggregate["records_with_unexplained_differences"]
    )

    return aggregate


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def _summary_lines(aggregate: dict) -> str:
    return "\n".join(
        [
            f"records            : {len(aggregate['record_names'])} "
            f"({', '.join(aggregate['record_names'])})",
            f"samples accepted   : {aggregate['total_samples_accepted']} / "
            f"{aggregate['total_input_samples']}",
            f"peaks              : whole {aggregate['total_whole_record_peaks']} | "
            f"streaming {aggregate['total_streaming_peaks']} | "
            f"exact {aggregate['total_exact_peak_matches']} | "
            f"missing {aggregate['total_missing_peaks']} | "
            f"extra {aggregate['total_extra_peaks']} | "
            f"max offset {aggregate['largest_absolute_peak_offset']}",
            f"sequences          : expected "
            f"{aggregate['total_offline_targets_expected']} | "
            f"emitted {aggregate['total_streaming_sequences_emitted']} | "
            f"exact {aggregate['total_exactly_matched_sequences']} | "
            f"missing {aggregate['total_missing_targets']} | "
            f"extra {aggregate['total_extra_targets']}",
            f"content mismatches : ecg {aggregate['total_ecg_window_mismatches']} | "
            f"rr {aggregate['total_rr_feature_mismatches']}",
            f"explained          : deployment-only "
            f"{aggregate['total_expected_deployment_only_targets']} | "
            f"causal-only {aggregate['total_causal_detector_only_targets']} | "
            f"causal missing "
            f"{aggregate['total_missing_targets_due_to_causal_divergence']}",
            f"unexplained        : extra "
            f"{aggregate['total_unexplained_extra_targets']} | "
            f"missing {aggregate['total_unexplained_missing_targets']} | "
            f"content {aggregate['total_unexplained_content_mismatches']}",
            f"exact parity       : {aggregate['all_records_exact_parity']}",
            f"all explained      : {aggregate['all_records_differences_explained']}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare streaming output with the offline XQRS dataset."
    )
    parser.add_argument("--record-name", type=str, default=DEFAULT_RECORD_NAME)
    parser.add_argument(
        "--all-validation-records",
        action="store_true",
        help="Evaluate every record in the validation split.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Plot detection divergence for records whose peaks disagree.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Silence PhysioNet/WFDB HTTP request logs.
    logging.getLogger("wfdb.io._url").setLevel(logging.WARNING)

    args = parse_args()

    if args.all_validation_records:
        record_names = load_split_record_names(args.split_summary_path, "val")
    else:
        record_names = [args.record_name]

    record_results: list[dict] = []
    failed_records: list[str] = []

    for record_name in record_names:
        try:
            result = evaluate_record(
                record_name=record_name,
                chunk_size=args.chunk_size,
                write_diagnostics=args.write_diagnostics,
            )
        except Exception:
            failed_records.append(record_name)
            logger.exception("Record %s could not be evaluated", record_name)
            continue

        record_results.append(result)
        _write_json(result, args.output_dir / f"record_{record_name}.json")

    aggregate = aggregate_records(record_results, failed_records)
    _write_json(aggregate, args.output_dir / "streaming_parity_summary.json")

    print(_summary_lines(aggregate))

    # The verdict already accounts for failed records and broken streams.
    if not aggregate["all_records_differences_explained"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
