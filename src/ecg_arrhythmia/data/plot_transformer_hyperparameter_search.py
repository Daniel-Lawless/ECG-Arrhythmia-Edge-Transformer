import matplotlib.pyplot as plt
import numpy as np

from ecg_arrhythmia.visualisation.matrix_plots import MODEL_EVALUATION_FIGURES_DIR

EXPERIMENT_NAMES = [
    "Baseline",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
]

BEST_VAL_MACRO_F1 = [
    0.5940,
    0.6466,
    0.6204,
    0.6546,
    0.6624,
    0.6160,
    0.6901,
    0.6750,
    0.6806,
    0.6714,
]


def main() -> None:
    figures_dir = MODEL_EVALUATION_FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(EXPERIMENT_NAMES))

    bar_colours = [
        "lightcoral" if name != "F" else "tab:red" for name in EXPERIMENT_NAMES
    ]

    plt.figure(figsize=(10, 5))

    bars = plt.bar(
        x,
        BEST_VAL_MACRO_F1,
        color=bar_colours,
        edgecolor="black",
        linewidth=1,
    )

    plt.bar_label(
        bars,
        fmt="%.4f",
        padding=3,
    )

    plt.xticks(x, EXPERIMENT_NAMES)
    plt.ylim(0.0, 0.74)
    plt.ylabel("Best Validation Macro F1")
    plt.title("Transformer Hyperparameter Search")

    plt.tight_layout()
    plt.savefig(
        figures_dir / "transformer_hyperparameter_search.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved hyperparameter search figure to:", figures_dir)


if __name__ == "__main__":
    main()
