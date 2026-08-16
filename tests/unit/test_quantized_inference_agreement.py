import numpy as np
import pytest

from ecg_arrhythmia.evaluation.evaluate_quantized_inference_agreement import (
    build_aggregate,
    classify_with_both,
    compare_record_arrays,
    disagreement_entries,
    logit_margins,
    transition_counts,
)
from ecg_arrhythmia.evaluation.quantization_agreement_plots import (
    write_aggregate_agreement_figures,
    write_record_agreement_figures,
)

# Four sequences: N, S, V, F under FP32.
FP32_LOGITS = np.array(
    [
        [4.0, 1.0, 0.0, -1.0],
        [0.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 5.0, 2.0],
        [1.0, 0.0, 2.0, 6.0],
    ],
    dtype=np.float32,
)
TARGET_PEAKS = np.array([1000, 2000, 3000, 4000], dtype=np.int64)


def _int8_with_one_flip() -> np.ndarray:
    """INT8 logits agreeing everywhere except sequence 1: S becomes N."""

    int8 = FP32_LOGITS + np.float32(0.01)
    int8[1] = [3.5, 3.0, 1.0, 0.0]

    return int8


# ---------------------------------------------------------------------
#                        Agreement And Drift
# ---------------------------------------------------------------------


def test_identical_logits_agree_everywhere():
    summary, arrays = compare_record_arrays(
        FP32_LOGITS,
        FP32_LOGITS.copy(),
        TARGET_PEAKS,
    )

    assert summary["num_sequences_compared"] == 4
    assert summary["class_agreements"] == 4
    assert summary["class_disagreements"] == 0
    assert summary["class_agreement_percentage"] == 100.0
    assert summary["mean_absolute_logit_difference"] == 0.0
    assert summary["maximum_absolute_logit_difference"] == 0.0
    assert summary["disagreement_target_peaks"] == []
    assert summary["fp32_logit_margin_mean_disagreeing"] is None
    assert arrays["agreed"].all()


def test_a_flipped_class_is_counted_and_traceable():
    summary, _ = compare_record_arrays(
        FP32_LOGITS,
        _int8_with_one_flip(),
        TARGET_PEAKS,
    )

    assert summary["class_agreements"] == 3
    assert summary["class_disagreements"] == 1
    assert summary["class_agreement_percentage"] == 75.0
    assert summary["class_disagreement_percentage"] == 25.0
    assert summary["disagreement_target_peaks"] == [2000]


def test_the_agreement_matrix_has_fp32_rows_and_int8_columns():
    summary, _ = compare_record_arrays(
        FP32_LOGITS,
        _int8_with_one_flip(),
        TARGET_PEAKS,
    )
    matrix = summary["agreement_matrix"]

    # FP32 predicted S (row 1); INT8 predicted N (column 0).
    assert matrix[1][0] == 1
    assert matrix[0][0] == 1
    assert matrix[2][2] == 1
    assert matrix[3][3] == 1
    assert sum(sum(row) for row in matrix) == 4


def test_logit_drift_statistics_are_computed_from_known_values():
    int8 = FP32_LOGITS.copy()
    int8[0, 0] += 0.4
    int8[2, 3] += 0.1

    summary, arrays = compare_record_arrays(FP32_LOGITS, int8, TARGET_PEAKS)

    assert summary["maximum_absolute_logit_difference"] == pytest.approx(0.4)
    # Two non-zero differences over sixteen logit values.
    assert summary["mean_absolute_logit_difference"] == pytest.approx(0.5 / 16)

    # rtol reflects float32 arithmetic: 0.4 is not exactly representable.
    np.testing.assert_allclose(
        arrays["per_sequence_max"],
        [0.4, 0.0, 0.1, 0.0],
        rtol=1e-6,
    )
    assert summary[
        "mean_per_sequence_maximum_absolute_logit_difference"
    ] == pytest.approx(0.125)
    assert summary[
        "median_per_sequence_maximum_absolute_logit_difference"
    ] == pytest.approx(0.05)
    # Linear interpolation over [0, 0, 0.1, 0.4].
    assert summary[
        "p95_per_sequence_maximum_absolute_logit_difference"
    ] == pytest.approx(0.355)


def test_margins_are_winning_minus_second_highest():
    margins = logit_margins(FP32_LOGITS)

    np.testing.assert_allclose(margins, [3.0, 2.0, 3.0, 4.0])


def test_margin_means_are_split_by_agreement():
    summary, _ = compare_record_arrays(
        FP32_LOGITS,
        _int8_with_one_flip(),
        TARGET_PEAKS,
    )

    # Agreeing sequences have FP32 margins 3, 3 and 4.
    assert summary["fp32_logit_margin_mean_agreeing"] == pytest.approx(10 / 3)
    # The flipped sequence had the smallest FP32 margin.
    assert summary["fp32_logit_margin_mean_disagreeing"] == pytest.approx(2.0)


# ---------------------------------------------------------------------
#                            Transitions
# ---------------------------------------------------------------------


def test_all_off_diagonal_transitions_are_generated():
    matrix = [
        [10, 2, 0, 0],
        [1, 5, 0, 0],
        [0, 0, 7, 0],
        [0, 0, 0, 3],
    ]

    counts = transition_counts(matrix)

    assert len(counts) == 12
    assert counts["N_to_S"] == 2
    assert counts["S_to_N"] == 1
    assert counts["V_to_F"] == 0
    assert "N_to_N" not in counts


# ---------------------------------------------------------------------
#                          Input Validation
# ---------------------------------------------------------------------


def test_mismatched_logit_shapes_are_rejected():
    with pytest.raises(ValueError, match="shapes must match"):
        compare_record_arrays(FP32_LOGITS, FP32_LOGITS[:2], TARGET_PEAKS)


def test_empty_inputs_are_rejected():
    empty = np.zeros((0, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="At least one sequence"):
        compare_record_arrays(empty, empty, np.zeros(0, dtype=np.int64))


def test_a_missing_target_peak_is_rejected():
    with pytest.raises(ValueError, match="One target peak per sequence"):
        compare_record_arrays(FP32_LOGITS, FP32_LOGITS.copy(), TARGET_PEAKS[:3])


# ---------------------------------------------------------------------
#                      Identical Input Ordering
# ---------------------------------------------------------------------


class RecordingClassifier:
    """Classifier stand-in that records exactly what it was asked to predict."""

    def __init__(self) -> None:
        self.seen: list = []

    def predict(self, sequence):
        self.seen.append(sequence)

        return sequence


def test_both_classifiers_receive_the_same_objects_in_the_same_order():
    sequences = [object() for _ in range(5)]
    fp32 = RecordingClassifier()
    int8 = RecordingClassifier()

    classify_with_both(sequences, fp32, int8)

    # Identity, not equality: the very same BeatSequence objects.
    assert all(seen is seq for seen, seq in zip(fp32.seen, sequences, strict=True))
    assert all(seen is seq for seen, seq in zip(int8.seen, sequences, strict=True))


def test_classifying_no_sequences_is_rejected():
    with pytest.raises(ValueError, match="At least one sequence"):
        classify_with_both([], RecordingClassifier(), RecordingClassifier())


# ---------------------------------------------------------------------
#                       Disagreement Entries
# ---------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, target_peak_index, label, logits):
        self.target_peak_index = target_peak_index
        self.peak_indices = (1, 2, 3, 4, target_peak_index)
        self.predicted_label = label
        self.logits = np.asarray(logits, dtype=np.float32)


def test_disagreement_entries_contain_both_sides():
    int8_logits = _int8_with_one_flip()
    _, arrays = compare_record_arrays(FP32_LOGITS, int8_logits, TARGET_PEAKS)

    fp32_events = [
        _FakeEvent(int(peak), label, logits)
        for peak, label, logits in zip(
            TARGET_PEAKS, ["N", "S", "V", "F"], FP32_LOGITS, strict=True
        )
    ]
    int8_events = [
        _FakeEvent(int(peak), label, logits)
        for peak, label, logits in zip(
            TARGET_PEAKS, ["N", "N", "V", "F"], int8_logits, strict=True
        )
    ]

    [entry] = disagreement_entries("114", fp32_events, int8_events, arrays)

    assert entry["record_name"] == "114"
    assert entry["target_peak_index"] == 2000
    assert entry["fp32_predicted_label"] == "S"
    assert entry["int8_predicted_label"] == "N"
    assert entry["fp32_logits"] == [float(v) for v in FP32_LOGITS[1]]
    assert entry["int8_logits"] == [float(v) for v in int8_logits[1]]
    assert entry["maximum_absolute_logit_difference"] == pytest.approx(3.5)
    assert entry["fp32_logit_margin"] == pytest.approx(2.0)


# ---------------------------------------------------------------------
#                        Aggregate Weighting
# ---------------------------------------------------------------------


def _record_result(name, num_sequences, agreements, mean_abs, max_abs, matrix):
    return {
        "record_name": name,
        "num_sequences_compared": num_sequences,
        "class_agreements": agreements,
        "class_disagreements": num_sequences - agreements,
        "mean_absolute_logit_difference": mean_abs,
        "maximum_absolute_logit_difference": max_abs,
        "agreement_matrix": matrix,
    }


def test_aggregates_pool_rather_than_average_record_percentages():
    diagonal = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    mixed = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    # Record A: 1 sequence, all agreeing. Record B: 3 sequences, 2 agreeing.
    # Equal-weight averaging would give 83.33%; pooling gives 75%.
    aggregate = build_aggregate(
        record_results=[
            _record_result("114", 1, 1, 0.1, 0.2, diagonal),
            _record_result("122", 3, 2, 0.5, 0.9, mixed),
        ],
        per_sequence_max_arrays=[
            np.array([0.2], dtype=np.float32),
            np.array([0.1, 0.5, 0.9], dtype=np.float32),
        ],
        chunk_size=36,
    )

    assert aggregate["total_sequences_compared"] == 4
    assert aggregate["class_agreement_percentage"] == pytest.approx(75.0)
    # Pooled mean: (0.1 * 1 + 0.5 * 3) / 4, not (0.1 + 0.5) / 2.
    assert aggregate["mean_absolute_logit_difference"] == pytest.approx(0.4)
    assert aggregate["maximum_absolute_logit_difference"] == pytest.approx(0.9)

    # Pooled percentiles come from the concatenated per-sequence values.
    pooled = np.array([0.2, 0.1, 0.5, 0.9])
    assert aggregate[
        "p95_per_sequence_maximum_absolute_logit_difference"
    ] == pytest.approx(float(np.percentile(pooled, 95)))
    assert aggregate[
        "median_per_sequence_maximum_absolute_logit_difference"
    ] == pytest.approx(float(np.percentile(pooled, 50)))

    # Matrices sum elementwise, and transitions come from the summed matrix.
    assert aggregate["agreement_matrix"][1][0] == 1
    assert aggregate["transition_counts"]["S_to_N"] == 1
    assert aggregate["records_with_class_disagreements"] == ["122"]
    assert aggregate["number_of_records_with_class_disagreements"] == 1


def test_aggregate_requires_one_array_per_record():
    with pytest.raises(ValueError, match="per record result"):
        build_aggregate(
            record_results=[_record_result("114", 1, 1, 0.0, 0.0, [[0] * 4] * 4)],
            per_sequence_max_arrays=[],
            chunk_size=36,
        )


def test_an_empty_aggregate_is_well_formed():
    aggregate = build_aggregate([], [], chunk_size=36, failed_records=["114"])

    assert aggregate["num_records_evaluated"] == 0
    assert aggregate["total_sequences_compared"] == 0
    assert aggregate["class_agreement_percentage"] == 0.0
    assert aggregate["failed_records"] == ["114"]


# ---------------------------------------------------------------------
#                                Plots
# ---------------------------------------------------------------------


def test_record_figures_render_for_a_tiny_example(tmp_path):
    int8_logits = _int8_with_one_flip()
    summary, arrays = compare_record_arrays(FP32_LOGITS, int8_logits, TARGET_PEAKS)

    written = write_record_agreement_figures(
        record_name="synthetic",
        fp32_logits=FP32_LOGITS,
        int8_logits=int8_logits,
        target_peaks=TARGET_PEAKS,
        matrix=summary["agreement_matrix"],
        fp32_margins=arrays["fp32_margins"],
        agreed=arrays["agreed"],
        figures_dir=tmp_path,
    )

    # Matrix, scatter, histogram, across-record, and the margin figure
    # because this example contains a disagreement.
    assert len(written) == 5
    assert all(path.exists() for path in written)
    assert {path.suffix for path in tmp_path.iterdir()} == {".png"}


def test_aggregate_figures_render_for_a_tiny_example(tmp_path):
    written = write_aggregate_agreement_figures(
        matrix=[[2, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        pooled_per_sequence_max=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        figures_dir=tmp_path,
    )

    assert len(written) == 2
    assert all(path.exists() for path in written)
