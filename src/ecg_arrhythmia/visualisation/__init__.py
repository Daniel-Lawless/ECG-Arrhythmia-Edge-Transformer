"""
Shared figure drawing.

Neutral by design: modules here depend only on pathlib, NumPy and
matplotlib, so any part of the project can draw with them without
pulling in training, inference, deployment, evaluation or streaming
code.

    matrix_plots
        the row-normalised matrix used by both the ground-truth
        confusion matrices and the implementation-agreement matrices
"""
