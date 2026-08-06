import numpy as np

from ecg_arrhythmia.evaluation.evaluate_streaming_parity import (
    _compare_sequences,
    _is_exact_parity,
    _is_explained,
    _OfflineReference,
    _Sequence,
    _StreamingRun,
    aggregate_records,
)
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SEQUENCE_LENGTH,
    WINDOW_SIZE,
)

# Ten evenly spaced beats, so the last six can be sequence targets.
BEAT_PEAKS = np.arange(100, 1100, 100, dtype=np.int64)
TARGET_PEAKS = [int(peak) for peak in BEAT_PEAKS[SEQUENCE_LENGTH - 1 :]]


def _sequence(peaks, ecg_value: float = 1.0, rr_value: float = 1.0) -> _Sequence:
    return _Sequence(
        ecg=np.full((SEQUENCE_LENGTH, WINDOW_SIZE), ecg_value),
        rr=np.full((SEQUENCE_LENGTH, 2), rr_value),
        peaks=tuple(int(peak) for peak in peaks),
    )


def _offline(scored: set[int] | None = None) -> _OfflineReference:
    sequences = {
        int(BEAT_PEAKS[target]): _sequence(
            BEAT_PEAKS[target - SEQUENCE_LENGTH + 1 : target + 1]
        )
        for target in range(SEQUENCE_LENGTH - 1, BEAT_PEAKS.size)
    }

    return _OfflineReference(
        beat_peaks=BEAT_PEAKS,
        beat_index_of_peak={int(peak): i for i, peak in enumerate(BEAT_PEAKS)},
        sequences=sequences,
        scored_peaks=set(TARGET_PEAKS) if scored is None else set(scored),
    )


def _run(sequences: dict[int, _Sequence], peaks=None) -> _StreamingRun:
    return _StreamingRun(
        sequences=sequences,
        peaks=np.asarray(BEAT_PEAKS if peaks is None else peaks, dtype=np.int64),
        total_input_samples=1200,
        samples_accepted=1200,
        continuity_validated=True,
    )


# ---------------------------------------------------------------------
#                        Difference Classification
# ---------------------------------------------------------------------


def test_identical_output_reports_no_differences():
    offline = _offline()

    result = _compare_sequences(offline, _run(dict(offline.sequences)), BEAT_PEAKS)

    assert result["exactly_matched_sequences"] == len(TARGET_PEAKS)
    assert result["missing_targets"] == []
    assert result["extra_targets"] == []
    assert result["unexplained_content_mismatches"] == []


def test_an_unscored_offline_sequence_is_a_deployment_only_target():
    # The offline dataset filtered this target out because its detection
    # carries no supported expert label, but streaming still emits it.
    offline = _offline(scored=set(TARGET_PEAKS) - {1000})

    result = _compare_sequences(offline, _run(dict(offline.sequences)), BEAT_PEAKS)

    assert result["extra_targets"] == [1000]
    assert result["expected_deployment_only_targets"] == [1000]
    assert result["unexplained_extra_targets"] == []


def test_a_peak_only_the_causal_detector_found_is_classified_as_such():
    offline = _offline()
    streaming = dict(offline.sequences)
    streaming[1050] = _sequence([650, 750, 850, 950, 1050])

    result = _compare_sequences(
        offline,
        _run(streaming, peaks=[*BEAT_PEAKS.tolist(), 1050]),
        BEAT_PEAKS,
    )

    assert result["extra_targets"] == [1050]
    assert result["causal_detector_only_targets"] == [1050]
    assert result["unexplained_extra_targets"] == []


def test_a_target_the_causal_detector_missed_explains_a_missing_target():
    offline = _offline()
    streaming = {
        peak: sequence for peak, sequence in offline.sequences.items() if peak != 1000
    }

    result = _compare_sequences(
        offline,
        _run(streaming, peaks=BEAT_PEAKS[:-1]),
        BEAT_PEAKS,
    )

    assert result["missing_targets"] == [1000]
    assert result["missing_targets_due_to_causal_divergence"] == [1000]
    assert result["unexplained_missing_targets"] == []


def test_an_ecg_mismatch_is_explained_by_different_sequence_peaks():
    offline = _offline()
    streaming = dict(offline.sequences)
    streaming[700] = _sequence([250, 400, 500, 600, 700], ecg_value=9.0)

    result = _compare_sequences(offline, _run(streaming), BEAT_PEAKS)

    assert result["ecg_window_mismatches"] == [700]
    assert result["peak_index_mismatches"] == [700]
    assert result["ecg_mismatches_explained_by_peak_history"] == 1
    assert result["unexplained_content_mismatches"] == []


def test_an_rr_mismatch_is_explained_by_a_changed_detection_history():
    offline = _offline()
    streaming = dict(offline.sequences)
    streaming[700] = _sequence(offline.sequences[700].peaks, rr_value=2.0)

    # An extra detection inside the RR dependency history of target 700,
    # further back than the five beats the model itself sees.
    result = _compare_sequences(
        offline,
        _run(streaming, peaks=sorted([*BEAT_PEAKS.tolist(), 650])),
        BEAT_PEAKS,
    )

    assert result["rr_feature_mismatches"] == [700]
    assert result["rr_mismatches_explained_by_peak_history"] == 1
    assert result["unexplained_content_mismatches"] == []


def test_an_rr_mismatch_with_an_identical_history_is_unexplained():
    offline = _offline()
    streaming = dict(offline.sequences)
    streaming[700] = _sequence(offline.sequences[700].peaks, rr_value=2.0)

    result = _compare_sequences(offline, _run(streaming), BEAT_PEAKS)

    assert result["rr_mismatches_explained_by_peak_history"] == 0
    assert result["unexplained_content_mismatches"] == [700]


# ---------------------------------------------------------------------
#                        Aggregate Interpretation
# ---------------------------------------------------------------------

_LIST_FIELDS = (
    "missing_peaks",
    "extra_peaks",
    "missing_targets",
    "extra_targets",
    "peak_index_mismatches",
    "ecg_window_mismatches",
    "rr_feature_mismatches",
    "expected_deployment_only_targets",
    "causal_detector_only_targets",
    "missing_targets_due_to_causal_divergence",
    "unexplained_extra_targets",
    "unexplained_missing_targets",
    "unexplained_content_mismatches",
)


def _record_summary(record_name: str, **overrides) -> dict:
    """A synthetic per-record summary with no differences, plus overrides."""

    summary: dict = {
        "record_name": record_name,
        "sampling_rate": 360.0,
        "chunk_size": 36,
        "total_input_samples": 650_000,
        "samples_accepted": 650_000,
        "continuity_validated": True,
        "whole_record_peaks": 2000,
        "streaming_peaks": 2000,
        "exact_peak_matches": 2000,
        "largest_absolute_peak_offset": 0,
        "offline_targets_expected": 1990,
        "streaming_sequences_emitted": 1990,
        "exactly_matched_sequences": 1990,
        "ecg_mismatches_explained_by_peak_history": 0,
        "rr_mismatches_explained_by_peak_history": 0,
    }
    summary.update({field: [] for field in _LIST_FIELDS})
    summary.update(overrides)
    summary["exact_parity"] = _is_exact_parity(summary)
    summary["all_differences_explained"] = _is_explained(summary)

    return summary


def test_exact_parity_can_be_false_while_every_difference_is_explained():
    exact = _record_summary("114")
    diverged = _record_summary(
        "210",
        streaming_peaks=2004,
        extra_peaks=[10, 20, 30, 40],
        extra_targets=[50, 60, 70, 80],
        causal_detector_only_targets=[50, 60, 70, 80],
    )

    aggregate = aggregate_records([exact, diverged])

    assert aggregate["all_records_continuity_validated"] is True
    assert aggregate["all_records_exact_parity"] is False
    assert aggregate["all_records_differences_explained"] is True
    assert aggregate["records_with_perfect_sequence_parity"] == ["114"]
    assert aggregate["records_with_perfect_peak_parity"] == ["114"]
    assert aggregate["total_extra_targets"] == 4
    assert aggregate["total_causal_detector_only_targets"] == 4
    assert aggregate["total_streaming_peaks"] == 4004


def test_a_broken_stream_is_never_exact_however_few_mismatches_it_collected():
    summary = _record_summary("210", continuity_validated=False)

    # Nothing mismatched, but the comparison never saw the whole record.
    assert summary["missing_targets"] == []
    assert summary["rr_feature_mismatches"] == []
    assert summary["exact_parity"] is False
    assert summary["all_differences_explained"] is False


def test_a_broken_stream_fails_both_aggregate_verdicts():
    aggregate = aggregate_records(
        [_record_summary("114"), _record_summary("210", continuity_validated=False)]
    )

    assert aggregate["all_records_continuity_validated"] is False
    assert aggregate["all_records_exact_parity"] is False
    assert aggregate["all_records_differences_explained"] is False
    assert aggregate["records_with_unexplained_differences"] == ["210"]


def test_an_unexplained_difference_fails_the_aggregate():
    summary = _record_summary(
        "210",
        extra_targets=[50],
        unexplained_extra_targets=[50],
    )

    aggregate = aggregate_records([summary])

    assert aggregate["all_records_differences_explained"] is False
    assert aggregate["records_with_unexplained_differences"] == ["210"]


def test_a_failed_record_fails_the_aggregate():
    # 114 on its own is at exact parity, so only the unevaluated record
    # can be what makes both verdicts negative.
    aggregate = aggregate_records([_record_summary("114")], failed_records=["233"])

    assert aggregate["failed_records"] == ["233"]
    assert aggregate["all_records_exact_parity"] is False
    assert aggregate["all_records_differences_explained"] is False


def test_no_evaluated_records_gives_no_positive_verdict():
    aggregate = aggregate_records([])

    assert aggregate["num_records_evaluated"] == 0
    assert aggregate["all_records_exact_parity"] is False
    assert aggregate["all_records_differences_explained"] is False


def test_aggregate_reports_the_largest_offset_across_records():
    aggregate = aggregate_records(
        [
            _record_summary("114"),
            _record_summary("210", largest_absolute_peak_offset=3),
        ]
    )

    assert aggregate["largest_absolute_peak_offset"] == 3
    assert aggregate["num_records_evaluated"] == 2
    assert aggregate["chunk_size"] == 36
