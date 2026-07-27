from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from ecg_arrhythmia.data.build_expert_centered_dataset import (
    build_expert_centered_dataset,
    build_expert_record_beats,
    verify_matches_saved_split,
)
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    SAMPLING_RATE,
    WINDOW_SIZE,
)

LOAD_RECORD_TARGET = "ecg_arrhythmia.data.build_expert_centered_dataset.load_record"
SELECT_CHANNEL_TARGET = (
    "ecg_arrhythmia.data.build_expert_centered_dataset.select_signal_channel"
)

SIGNAL_LENGTH = 2000

# "+" is a non-heartbeat and "/" maps to the excluded Q class.
ANNOTATION_SAMPLES = [50, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
ANNOTATION_SYMBOLS = ["+", "N", "V", "/", "S", "N", "N", "N", "N"]


class FakeAnnotation:
    def __init__(self, sample: list[int], symbol: list[str]) -> None:
        self.sample = np.asarray(sample, dtype=np.int64)
        self.symbol = list(symbol)


@contextmanager
def _mocked_record():
    signal = np.arange(SIGNAL_LENGTH, dtype=np.float64)
    annotation = FakeAnnotation(ANNOTATION_SAMPLES, ANNOTATION_SYMBOLS)

    def fake_load_record(record_name):
        return None, {"fs": SAMPLING_RATE}, annotation

    def fake_select_signal_channel(signals, fields):
        return signal, "MLII"

    with (
        patch(LOAD_RECORD_TARGET, side_effect=fake_load_record),
        patch(SELECT_CHANNEL_TARGET, side_effect=fake_select_signal_channel),
    ):
        yield signal


def test_expert_beats_carry_identity_and_drop_q():
    with _mocked_record():
        beats = build_expert_record_beats(
            record_name="rec",
            normalise_beats=False,
            excluded_labels={"Q"},
        )

    # The seed beat (200) and the Q beat (600) are excluded; six beats remain.
    assert beats.num_beats == 6
    np.testing.assert_array_equal(
        beats.annotation_samples,
        np.array([400, 800, 1000, 1200, 1400, 1600]),
    )
    assert list(beats.aami_labels) == ["V", "S", "N", "N", "N", "N"]

    # Annotation indices refer to positions in the heartbeat-filtered list.
    np.testing.assert_array_equal(
        beats.annotation_indices,
        np.array([1, 3, 4, 5, 6, 7]),
    )


def test_expert_windows_are_centred_on_annotations():
    with _mocked_record():
        beats = build_expert_record_beats(
            record_name="rec",
            normalise_beats=False,
            excluded_labels={"Q"},
        )

    assert beats.windows.shape == (6, WINDOW_SIZE)

    # First kept beat is the V annotation at 400.
    np.testing.assert_array_equal(
        beats.windows[0],
        np.arange(400 - SAMPLES_BEFORE, 400 + SAMPLES_AFTER, dtype=np.float64),
    )


def test_expert_rr_keeps_q_beats_in_the_timeline():
    with _mocked_record():
        beats = build_expert_record_beats(
            record_name="rec",
            normalise_beats=False,
            excluded_labels={"Q"},
        )

    # The S beat at 800 follows the Q beat at 600 in the annotation timeline,
    # so its previous RR interval is (800 - 600) samples. If Q beats were
    # removed before RR calculation, it would be (800 - 400) samples instead.
    assert beats.rr_features[1, 0] == pytest.approx((800 - 600) / SAMPLING_RATE)
    assert beats.rr_features[1, 0] != pytest.approx((800 - 400) / SAMPLING_RATE)


def test_build_expert_dataset_targets_carry_identity():
    with _mocked_record():
        dataset = build_expert_centered_dataset(record_names=["rec"])

    # Six beats give two length-5 sequences, targeting the beats at 1400/1600.
    assert dataset.X_sequences.shape == (2, 5, WINDOW_SIZE)
    assert list(dataset.y_labels) == ["N", "N"]
    np.testing.assert_array_equal(
        dataset.target_annotation_samples,
        np.array([1400, 1600]),
    )
    np.testing.assert_array_equal(
        dataset.target_records,
        np.array(["rec", "rec"]),
    )


def test_verify_matches_saved_split(tmp_path):
    with _mocked_record():
        dataset = build_expert_centered_dataset(record_names=["rec"])

    saved_dir = tmp_path / "val"
    saved_dir.mkdir()
    np.save(saved_dir / "y.npy", np.array(["N", "N"]))
    np.save(saved_dir / "patient_ids.npy", np.array(["rec", "rec"]))

    # Matching supports and record counts pass verification.
    verify_matches_saved_split(dataset, saved_dir)

    # A different class support must fail loudly.
    np.save(saved_dir / "y.npy", np.array(["N", "S"]))
    with pytest.raises(ValueError, match="class supports"):
        verify_matches_saved_split(dataset, saved_dir)
