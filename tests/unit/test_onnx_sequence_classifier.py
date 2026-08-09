from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.data.label_mapping import NUM_CLASSES
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SEQUENCE_LENGTH,
    WINDOW_SIZE,
)
from ecg_arrhythmia.streaming import onnx_sequence_classifier as classifier_module
from ecg_arrhythmia.streaming.onnx_contract import (
    ONNX_ECG_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    ONNX_RR_INPUT_NAME,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.sequence_assembler import BeatSequence

MODEL_PATH = Path("model.onnx")
PEAK_INDICES = (500, 860, 1220, 1580, 1940)

ECG_SHAPE = ["batch_size", SEQUENCE_LENGTH, 1, WINDOW_SIZE]
RR_SHAPE = ["batch_size", SEQUENCE_LENGTH, 2]
LOGITS_SHAPE = ["batch_size", NUM_CLASSES]


class _ArgNode:
    """Stand-in for an ONNX Runtime NodeArg."""

    def __init__(self, name: str, shape: list, node_type: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = node_type


class FakeSession:
    """Minimal stand-in for ort.InferenceSession."""

    def __init__(
        self,
        logits=None,
        ecg_node: _ArgNode | None = None,
        rr_node: _ArgNode | None = None,
        logits_node: _ArgNode | None = None,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        self._logits = (
            np.array([[0.1, 2.5, 0.3, 0.4]], dtype=np.float32)
            if logits is None
            else np.asarray(logits)
        )
        self._ecg_node = ecg_node or _ArgNode(ONNX_ECG_INPUT_NAME, ECG_SHAPE)
        self._rr_node = rr_node or _ArgNode(ONNX_RR_INPUT_NAME, RR_SHAPE)
        self._logits_node = logits_node or _ArgNode(ONNX_OUTPUT_NAME, LOGITS_SHAPE)
        self._providers = providers
        self.run_calls: list[dict] = []

    def get_inputs(self) -> list[_ArgNode]:
        return [self._ecg_node, self._rr_node]

    def get_outputs(self) -> list[_ArgNode]:
        return [self._logits_node]

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def run(self, output_names, input_feed):
        self.run_calls.append(input_feed)
        return [self._logits]


@pytest.fixture
def build_classifier(monkeypatch):
    """Build a classifier over a fake session, counting session creations."""

    def build(session: FakeSession) -> tuple[ONNXSequenceClassifier, list[Path]]:
        created: list[Path] = []

        monkeypatch.setattr(
            classifier_module,
            "validate_onnx_model",
            lambda onnx_path: None,
        )

        def fake_create_onnx_session(onnx_path: Path) -> FakeSession:
            created.append(onnx_path)
            return session

        monkeypatch.setattr(
            classifier_module,
            "create_onnx_session",
            fake_create_onnx_session,
        )

        return ONNXSequenceClassifier(MODEL_PATH), created

    return build


def _sequence(ecg=None, rr=None) -> BeatSequence:
    return BeatSequence(
        ecg=(
            np.arange(SEQUENCE_LENGTH * WINDOW_SIZE, dtype=np.float64).reshape(
                SEQUENCE_LENGTH, 1, WINDOW_SIZE
            )
            if ecg is None
            else ecg
        ),
        rr=np.ones((SEQUENCE_LENGTH, 2), dtype=np.float64) if rr is None else rr,
        target_peak_index=PEAK_INDICES[-1],
        peak_indices=PEAK_INDICES,
    )


# ---------------------------------------------------------------------
#                          Session Lifecycle
# ---------------------------------------------------------------------


def test_the_session_is_created_once_and_reused(build_classifier):
    classifier, created = build_classifier(FakeSession())

    for _ in range(3):
        classifier.predict(_sequence())

    assert created == [MODEL_PATH]


def test_the_configured_providers_are_exposed(build_classifier):
    classifier, _ = build_classifier(FakeSession())

    assert classifier.providers == ("CPUExecutionProvider",)


# ---------------------------------------------------------------------
#                            Input Handling
# ---------------------------------------------------------------------


def test_inputs_are_batched_and_converted_to_float32(build_classifier):
    session = FakeSession()
    classifier, _ = build_classifier(session)

    classifier.predict(_sequence())

    [input_feed] = session.run_calls
    ecg = input_feed[ONNX_ECG_INPUT_NAME]
    rr = input_feed[ONNX_RR_INPUT_NAME]

    assert ecg.shape == (1, SEQUENCE_LENGTH, 1, WINDOW_SIZE)
    assert rr.shape == (1, SEQUENCE_LENGTH, 2)
    assert ecg.dtype == np.float32
    assert rr.dtype == np.float32
    assert ecg.flags.c_contiguous
    assert rr.flags.c_contiguous


def test_non_contiguous_inputs_are_converted_without_mutating_the_source(
    build_classifier,
):
    # A reversed view is float64 and not C-contiguous.
    source = np.arange(SEQUENCE_LENGTH * WINDOW_SIZE * 2, dtype=np.float64).reshape(
        SEQUENCE_LENGTH, 1, WINDOW_SIZE * 2
    )[:, :, ::2]
    original = source.copy()

    session = FakeSession()
    classifier, _ = build_classifier(session)

    classifier.predict(_sequence(ecg=source))

    [input_feed] = session.run_calls
    ecg = input_feed[ONNX_ECG_INPUT_NAME]

    assert ecg.flags.c_contiguous
    np.testing.assert_array_equal(ecg[0], source.astype(np.float32))
    np.testing.assert_array_equal(source, original)


def test_a_wrong_ecg_shape_is_rejected(build_classifier):
    classifier, _ = build_classifier(FakeSession())

    with pytest.raises(ValueError, match="ECG input must have shape"):
        classifier.predict(_sequence(ecg=np.zeros((SEQUENCE_LENGTH, WINDOW_SIZE))))


def test_a_wrong_rr_shape_is_rejected(build_classifier):
    classifier, _ = build_classifier(FakeSession())

    with pytest.raises(ValueError, match="RR input must have shape"):
        classifier.predict(_sequence(rr=np.zeros((SEQUENCE_LENGTH, 3))))


# ---------------------------------------------------------------------
#                          Prediction Contract
# ---------------------------------------------------------------------


def test_the_prediction_reports_the_argmax_class_and_label(build_classifier):
    session = FakeSession(logits=np.array([[0.1, 2.5, 0.3, 0.4]], dtype=np.float32))
    classifier, _ = build_classifier(session)

    event = classifier.predict(_sequence())

    assert event.predicted_class_index == 1
    assert event.predicted_label == "S"
    assert event.logits.shape == (NUM_CLASSES,)


def test_sequence_metadata_is_carried_through(build_classifier):
    classifier, _ = build_classifier(FakeSession())

    event = classifier.predict(_sequence())

    assert event.target_peak_index == PEAK_INDICES[-1]
    assert event.peak_indices == PEAK_INDICES


def test_event_logits_cannot_be_mutated(build_classifier):
    classifier, _ = build_classifier(FakeSession())

    event = classifier.predict(_sequence())

    assert not event.logits.flags.writeable

    with pytest.raises(ValueError):
        event.logits[0] = 99.0


def test_an_unexpected_output_shape_is_rejected(build_classifier):
    session = FakeSession(logits=np.zeros((2, NUM_CLASSES), dtype=np.float32))
    classifier, _ = build_classifier(session)

    with pytest.raises(ValueError, match="must have shape"):
        classifier.predict(_sequence())


# ---------------------------------------------------------------------
#                          Model Contract Checks
# ---------------------------------------------------------------------


def test_a_model_with_the_wrong_input_rank_is_rejected(build_classifier):
    session = FakeSession(
        ecg_node=_ArgNode(ONNX_ECG_INPUT_NAME, ["batch_size", SEQUENCE_LENGTH, 240])
    )

    with pytest.raises(ValueError, match="must have rank 4"):
        build_classifier(session)


def test_a_model_with_the_wrong_window_size_is_rejected(build_classifier):
    session = FakeSession(
        ecg_node=_ArgNode(ONNX_ECG_INPUT_NAME, ["batch_size", SEQUENCE_LENGTH, 1, 200])
    )

    with pytest.raises(ValueError, match="axis 3 must be 240"):
        build_classifier(session)


def test_a_model_with_a_non_float_input_is_rejected(build_classifier):
    session = FakeSession(
        rr_node=_ArgNode(ONNX_RR_INPUT_NAME, RR_SHAPE, node_type="tensor(double)")
    )

    with pytest.raises(ValueError, match="tensor\\(float\\)"):
        build_classifier(session)


def test_a_model_with_the_wrong_class_count_is_rejected(build_classifier):
    session = FakeSession(logits_node=_ArgNode(ONNX_OUTPUT_NAME, ["batch_size", 3]))

    with pytest.raises(ValueError, match="axis 1 must be 4"):
        build_classifier(session)


def test_a_symbolic_dimension_is_accepted(build_classifier):
    # A dynamo-exported graph may report axes symbolically. That is not a
    # contract violation, so it must not be rejected.
    session = FakeSession(
        ecg_node=_ArgNode(ONNX_ECG_INPUT_NAME, ["batch_size", "s1", 1, WINDOW_SIZE])
    )

    classifier, _ = build_classifier(session)

    assert classifier.predict(_sequence()).predicted_label == "S"
