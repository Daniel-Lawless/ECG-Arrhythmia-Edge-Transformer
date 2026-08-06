import importlib
from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.data import plot_transformer_hyperparameter_search
from ecg_arrhythmia.data.plot_final_results import plot_confusion_matrix
from ecg_arrhythmia.visualisation.matrix_plots import (
    MODEL_EVALUATION_FIGURES_DIR,
    plot_row_normalised_matrix,
    row_normalise,
)

LABELS = ["N", "S", "V", "F"]

# A realistic exact-agreement matrix: N dominates the record, but every
# class agrees completely with itself.
IMBALANCED_AGREEMENT = np.array(
    [
        [1673, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 48, 0],
        [0, 0, 0, 147],
    ]
)


# ---------------------------------------------------------------------
#                            Module Location
# ---------------------------------------------------------------------


def test_the_shared_plotting_module_lives_in_the_visualisation_package():
    # A circular import between the visualisation package and any of its
    # callers would surface here.
    for module_name in (
        "ecg_arrhythmia.visualisation.matrix_plots",
        "ecg_arrhythmia.data.plot_final_results",
        "ecg_arrhythmia.data.plot_transformer_hyperparameter_search",
    ):
        assert importlib.import_module(module_name) is not None


def test_the_old_root_level_plotting_module_is_gone():
    # One canonical location: no forwarding shim was left behind.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ecg_arrhythmia.plotting")


# ---------------------------------------------------------------------
#                          Figure Destinations
# ---------------------------------------------------------------------


def test_model_evaluation_figures_have_their_own_directory():
    assert MODEL_EVALUATION_FIGURES_DIR == Path("artifacts/figures/model_evaluation")

    # They are a subdirectory of the figures root, not the root itself,
    # so no model-evaluation figure is written loose any more.
    assert MODEL_EVALUATION_FIGURES_DIR.parent == Path("artifacts/figures")
    assert MODEL_EVALUATION_FIGURES_DIR != Path("artifacts/figures")


def test_a_generating_script_creates_its_figure_directory(tmp_path, monkeypatch):
    figures_dir = tmp_path / "model_evaluation"
    monkeypatch.setattr(
        plot_transformer_hyperparameter_search,
        "MODEL_EVALUATION_FIGURES_DIR",
        figures_dir,
    )

    plot_transformer_hyperparameter_search.main()

    assert figures_dir.is_dir()
    assert (figures_dir / "transformer_hyperparameter_search.png").exists()

    # The figure is the only thing written, and nothing lands beside it.
    assert [path.name for path in tmp_path.iterdir()] == ["model_evaluation"]


# ---------------------------------------------------------------------
#                          Row Normalisation
# ---------------------------------------------------------------------


def test_every_exact_diagonal_row_normalises_to_one_hundred_percent():
    percentages = row_normalise(IMBALANCED_AGREEMENT) * 100

    np.testing.assert_allclose(
        np.diag(percentages),
        [100.0, 100.0, 100.0, 100.0],
    )

    # Five sequences in full agreement score the same as 1,673, which is
    # the whole reason for normalising by row rather than by raw count.
    assert percentages[1, 1] == percentages[0, 0]


def test_off_diagonal_percentages_are_relative_to_their_own_row():
    matrix = np.array(
        [
            [3, 1, 0, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    percentages = row_normalise(matrix) * 100

    assert percentages[0, 0] == pytest.approx(75.0)
    assert percentages[0, 1] == pytest.approx(25.0)
    assert percentages[1, 1] == pytest.approx(50.0)
    assert percentages[1, 2] == pytest.approx(50.0)


def test_an_empty_row_stays_at_zero_without_dividing_by_zero():
    matrix = np.array([[0, 0], [1, 1]])

    with np.errstate(divide="raise", invalid="raise"):
        percentages = row_normalise(matrix) * 100

    np.testing.assert_array_equal(percentages[0], [0.0, 0.0])
    assert not np.isnan(percentages).any()


def test_the_shared_helper_renders_a_small_matrix(tmp_path):
    output_path = tmp_path / "matrix.png"

    plot_row_normalised_matrix(
        matrix=IMBALANCED_AGREEMENT,
        labels=LABELS,
        title="Synthetic agreement",
        output_path=output_path,
        cmap="Blues",
        x_label="Comparison prediction",
        y_label="Reference prediction",
    )

    assert output_path.exists()


def test_plot_confusion_matrix_still_renders_from_a_saved_summary(tmp_path):
    summary = {
        "confusion_matrix": {
            "labels": LABELS,
            "rows": [
                {
                    "true_label": label,
                    "predictions": dict(
                        zip(LABELS, IMBALANCED_AGREEMENT[index], strict=True)
                    ),
                }
                for index, label in enumerate(LABELS)
            ],
        }
    }
    output_path = tmp_path / "confusion.png"

    plot_confusion_matrix(
        summary=summary,
        title="Synthetic confusion matrix",
        output_path=output_path,
        cmap="Blues",
    )

    assert output_path.exists()
