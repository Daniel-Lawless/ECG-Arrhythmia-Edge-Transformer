import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_edge_inference_does_not_load_training_or_dashboard_dependencies():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    program = """
import importlib.abc
import sys
from pathlib import Path

forbidden = {'torch', 'onnx', 'neurokit2', 'streamlit', 'plotly'}
class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError(f'edge tried to import {fullname}')
sys.meta_path.insert(0, RejectHeavyImports())

import numpy as np
import ecg_arrhythmia.transport.control_server
import ecg_arrhythmia.telemetry.live
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence

classifier = ONNXSequenceClassifier(Path(sys.argv[1]))
prediction = classifier.predict(BeatSequence(
    ecg=np.zeros((5, 1, 240), dtype=np.float32),
    rr=np.ones((5, 2), dtype=np.float32),
    target_peak_index=1800,
    peak_indices=(360, 720, 1080, 1440, 1800),
))
assert classifier.providers == ('CPUExecutionProvider',)
assert prediction.logits.shape == (4,)
assert np.isfinite(prediction.logits).all()
leaked = forbidden.intersection(name.split('.')[0] for name in sys.modules)
assert not leaked, f'edge imports pulled in {leaked}'
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(ROOT / "artifacts/models/ecg_sequence_transformer.onnx"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_packaging_keeps_runtime_extras_separate_and_native_dev_complete():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]

    assert project["dependencies"] == ["numpy"]
    assert extras["edge"] == ["wfdb", "onnxruntime==1.28.0"]
    assert extras["dashboard"] == ["streamlit>=1.61", "plotly>=5.18"]
    assert "ecg-arrhythmia[research]" in extras["dev"]
    assert "ecg-arrhythmia[edge]" in extras["research"]
    assert {"torch", "onnx", "neurokit2", "scikit-learn"} <= set(extras["research"])
