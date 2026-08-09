import sys
from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.data.label_mapping import NUM_CLASSES
from ecg_arrhythmia.evaluation.evaluate_streaming_inference_parity import (
    COMPARISON_PAIRS,
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_DIR,
    _write_record_logits,
    aggregate_records,
    agreement_matrix,
    compare_logits,
    parse_args,
)
from ecg_arrhythmia.evaluation.streaming_inference_plots import (
    per_sequence_maximum_difference,
    write_record_figures,
)

# Two sequences whose argmax classes are N and S.
REFERENCE = np.array(
    [
        [3.0, 1.0, 0.0, -1.0],
        [0.0, 2.0, 1.0, 0.5],
    ],
    dtype=np.float32,
)
TARGET_PEAKS = np.array([1000, 2000], dtype=np.int64)


# ---------------------------------------------------------------------
#                             Output Paths
# ---------------------------------------------------------------------


def test_results_default_to_the_deployment_evaluation_directory():
    assert DEFAULT_OUTPUT_DIR == Path(
        "artifacts/results/deployment_evaluation/streaming_inference_parity"
    )


def test_figures_still_default_to_their_own_directory():
    assert DEFAULT_FIGURES_DIR == Path("artifacts/figures/streaming_inference_parity")


def test_a_custom_output_directory_overrides_the_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_streaming_inference_parity",
            "--output-dir",
            "custom/results",
        ],
    )

    arguments = parse_args()

    assert arguments.output_dir == Path("custom/results")
    assert arguments.figures_dir == DEFAULT_FIGURES_DIR


def test_record_logits_are_saved_with_the_results(tmp_path):
    output_dir = tmp_path / "results"

    _write_record_logits(
        record_name="synthetic",
        logits={
            "pytorch": REFERENCE,
            "offline_onnx": REFERENCE.copy(),
            "streaming_onnx": REFERENCE.copy(),
        },
        target_peaks=TARGET_PEAKS,
        output_dir=output_dir,
    )

    logits_path = output_dir / "record_synthetic_logits.npz"
    assert logits_path.exists()

    with np.load(logits_path) as saved:
        assert set(saved) == {
            "pytorch_logits",
            "offline_onnx_logits",
            "streaming_onnx_logits",
            "target_peaks",
        }
        np.testing.assert_array_equal(saved["pytorch_logits"], REFERENCE)
        np.testing.assert_array_equal(saved["target_peaks"], TARGET_PEAKS)


# ---------------------------------------------------------------------
#                           Logit Comparison
# ---------------------------------------------------------------------


def test_identical_logits_pass_with_exact_equality():
    result = compare_logits(REFERENCE, REFERENCE.copy(), TARGET_PEAKS)

    assert result["num_sequences_compared"] == 2
    assert result["num_class_agreements"] == 2
    assert result["class_agreement_percentage"] == 100.0
    assert result["maximum_absolute_logit_difference"] == 0.0
    assert result["mean_absolute_logit_difference"] == 0.0
    assert result["arrays_exactly_equal"] is True
    assert result["num_arrays_outside_tolerance"] == 0
    assert result["passed"] is True


def test_a_tiny_difference_passes_tolerance_without_being_exactly_equal():
    comparison = REFERENCE + np.float32(1e-6)

    result = compare_logits(REFERENCE, comparison, TARGET_PEAKS)

    assert result["arrays_exactly_equal"] is False
    assert result["num_arrays_within_tolerance"] == 2
    assert result["maximum_absolute_logit_difference"] > 0.0
    assert result["passed"] is True


def test_a_class_disagreement_fails_and_is_traceable():
    comparison = REFERENCE.copy()
    # The first sequence flips from N to S.
    comparison[0] = [0.0, 5.0, 0.0, 0.0]

    result = compare_logits(REFERENCE, comparison, TARGET_PEAKS)

    assert result["num_class_agreements"] == 1
    assert result["class_agreement_percentage"] == 50.0
    assert result["class_disagreement_target_peaks"] == [1000]
    assert result["passed"] is False

    matrix = result["agreement_matrix"]
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1


def test_a_difference_outside_tolerance_fails_even_with_the_same_class():
    comparison = REFERENCE.copy()
    # Large enough to break tolerance, too small to change the argmax.
    comparison[1, 3] += 0.5

    result = compare_logits(REFERENCE, comparison, TARGET_PEAKS)

    assert result["num_class_agreements"] == 2
    assert result["num_arrays_outside_tolerance"] == 1
    assert result["tolerance_failure_target_peaks"] == [2000]
    assert result["passed"] is False


def test_mean_and_maximum_differences_are_measured():
    comparison = REFERENCE.copy()
    comparison[0, 0] += 0.8
    comparison[1, 1] += 0.4

    result = compare_logits(REFERENCE, comparison, TARGET_PEAKS)

    assert result["maximum_absolute_logit_difference"] == pytest.approx(0.8)
    # Two non-zero differences spread across eight logit values.
    assert result["mean_absolute_logit_difference"] == pytest.approx(1.2 / 8)


def test_mismatched_logit_shapes_are_rejected():
    with pytest.raises(ValueError, match="Logit shapes must match"):
        compare_logits(REFERENCE, REFERENCE[:1], TARGET_PEAKS[:1])


def test_agreement_matrix_places_disagreements_off_the_diagonal():
    matrix = agreement_matrix(np.array([0, 1, 2]), np.array([0, 2, 2]))

    assert matrix[0][0] == 1
    assert matrix[1][2] == 1
    assert matrix[2][2] == 1
    assert sum(sum(row) for row in matrix) == 3


# ---------------------------------------------------------------------
#                        Aggregate Interpretation
# ---------------------------------------------------------------------


def _comparison(
    num_sequences: int = 10,
    passed: bool = True,
    mean_difference: float = 0.0,
    maximum_difference: float = 0.0,
) -> dict:
    agreements = num_sequences if passed else num_sequences - 1

    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    matrix[0][0] = agreements
    matrix[0][1] = num_sequences - agreements

    return {
        "num_sequences_compared": num_sequences,
        "num_class_agreements": agreements,
        "class_agreement_percentage": agreements / num_sequences * 100.0,
        "mean_absolute_logit_difference": mean_difference,
        "maximum_absolute_logit_difference": maximum_difference,
        "num_arrays_within_tolerance": num_sequences if passed else 0,
        "num_arrays_outside_tolerance": 0 if passed else num_sequences,
        "arrays_exactly_equal": passed,
        "relative_tolerance": 1e-5,
        "absolute_tolerance": 1e-5,
        "class_disagreement_target_peaks": [] if passed else [1234],
        "tolerance_failure_target_peaks": [] if passed else [1234],
        "agreement_matrix": matrix,
        "passed": passed,
    }


def _record_summary(
    record_name: str,
    num_sequences: int = 10,
    failing_comparison: str | None = None,
    mean_difference: float = 0.0,
    maximum_difference: float = 0.0,
) -> dict:
    comparisons = {
        name: _comparison(
            num_sequences=num_sequences,
            passed=name != failing_comparison,
            mean_difference=mean_difference,
            maximum_difference=maximum_difference,
        )
        for name, _, _ in COMPARISON_PAIRS
    }

    return {
        "record_name": record_name,
        "chunk_size": 36,
        "num_sequences_compared": num_sequences,
        "first_target_peak": 1000,
        "last_target_peak": 9000,
        "relative_tolerance": 1e-5,
        "absolute_tolerance": 1e-5,
        "comparisons": comparisons,
        "parity_passed": all(
            comparison["passed"] for comparison in comparisons.values()
        ),
    }


def test_matching_records_aggregate_to_a_passing_verdict():
    aggregate = aggregate_records(
        [_record_summary("114"), _record_summary("122", num_sequences=6)]
    )

    assert aggregate["num_records_evaluated"] == 2
    assert aggregate["total_sequences_compared"] == 16
    assert aggregate["records_failing_parity"] == []
    assert aggregate["all_records_parity_passed"] is True

    for comparison in aggregate["comparisons"].values():
        assert comparison["total_sequences_compared"] == 16
        assert comparison["class_agreement_percentage"] == 100.0
        assert comparison["agreement_matrix"][0][0] == 16
        assert comparison["passed"] is True


def test_one_failing_comparison_fails_the_whole_verdict():
    aggregate = aggregate_records(
        [
            _record_summary("114"),
            _record_summary("210", failing_comparison="pytorch_vs_streaming_onnx"),
        ]
    )

    assert aggregate["records_failing_parity"] == ["210"]
    assert aggregate["all_records_parity_passed"] is False
    assert aggregate["comparisons"]["pytorch_vs_streaming_onnx"]["passed"] is False
    assert aggregate["comparisons"]["pytorch_vs_offline_onnx"]["passed"] is True


def test_a_failed_record_fails_the_verdict():
    aggregate = aggregate_records([_record_summary("114")], failed_records=["233"])

    assert aggregate["failed_records"] == ["233"]
    assert aggregate["all_records_parity_passed"] is False


def test_no_records_gives_no_positive_verdict():
    aggregate = aggregate_records([])

    assert aggregate["num_records_evaluated"] == 0
    assert aggregate["all_records_parity_passed"] is False


def test_mean_difference_is_weighted_by_sequence_count():
    aggregate = aggregate_records(
        [
            _record_summary("114", num_sequences=10, mean_difference=0.2),
            _record_summary(
                "122",
                num_sequences=30,
                mean_difference=0.6,
                maximum_difference=0.9,
            ),
        ]
    )
    comparison = aggregate["comparisons"]["pytorch_vs_streaming_onnx"]

    # (0.2 * 10 + 0.6 * 30) / 40
    assert comparison["mean_absolute_logit_difference"] == pytest.approx(0.5)
    assert comparison["maximum_absolute_logit_difference"] == pytest.approx(0.9)


# ---------------------------------------------------------------------
#                                Plots
# ---------------------------------------------------------------------


def test_per_sequence_maximum_difference_takes_the_worst_class():
    reference = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    comparison = np.array([[0.5, 0.1], [0.0, 0.25]], dtype=np.float32)

    np.testing.assert_allclose(
        per_sequence_maximum_difference(reference, comparison),
        [0.5, 0.25],
    )


def test_every_figure_renders_for_a_tiny_example(tmp_path):
    comparisons = {name: _comparison() for name, _, _ in COMPARISON_PAIRS}

    written = write_record_figures(
        record_name="synthetic",
        pytorch_logits=REFERENCE,
        streaming_onnx_logits=REFERENCE.copy(),
        target_peaks=TARGET_PEAKS,
        comparisons=comparisons,
        figures_dir=tmp_path,
    )

    # Three agreement matrices plus the three numerical parity figures.
    assert len(written) == 6
    assert all(path.exists() for path in written)

    # The figures directory holds figures and nothing else; the logits
    # archive is result data and lives with the JSON summaries.
    assert {path.suffix for path in tmp_path.iterdir()} == {".png"}
