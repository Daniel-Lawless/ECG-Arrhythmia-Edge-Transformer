import json

import pytest

from ecg_arrhythmia.evaluation.edge_sustained_resource_plots import (
    default_result_json,
    load_record_segments,
    record_segments,
    transition_minutes,
)

# One MIT-BIH record: 650,000 samples at 360 Hz.
RECORD_SECONDS = 1805.56


def _per_record(*durations_with_names) -> list[dict]:
    return [
        {"record_name": name, "record_wall_seconds": duration}
        for name, duration in durations_with_names
    ]


def test_segments_accumulate_record_durations_and_convert_to_minutes():
    segments = record_segments(
        _per_record(
            ("114", RECORD_SECONDS),
            ("122", RECORD_SECONDS),
            ("209", RECORD_SECONDS),
        )
    )

    # Cumulative ends at 1805.56, 3611.12 and 5416.68 seconds.
    assert [segment["end_minutes"] for segment in segments] == [
        pytest.approx(1805.56 / 60),
        pytest.approx(3611.12 / 60),
        pytest.approx(5416.68 / 60),
    ]
    assert segments[0]["start_minutes"] == pytest.approx(0.0)
    assert segments[1]["start_minutes"] == pytest.approx(1805.56 / 60)


def test_no_transition_at_time_zero_or_after_the_final_segment():
    segments = record_segments(
        _per_record(
            ("114", RECORD_SECONDS),
            ("122", RECORD_SECONDS),
            ("209", RECORD_SECONDS),
        )
    )

    transitions = transition_minutes(segments)

    # Interior boundaries only: two transitions for three records.
    assert transitions == [
        pytest.approx(1805.56 / 60),
        pytest.approx(3611.12 / 60),
    ]
    assert all(boundary > 0.0 for boundary in transitions)
    assert max(transitions) < segments[-1]["end_minutes"]


def test_a_single_record_produces_no_transitions():
    segments = record_segments(_per_record(("114", RECORD_SECONDS)))

    assert transition_minutes(segments) == []


def test_labels_sit_at_the_midpoint_of_their_interval():
    segments = record_segments(
        _per_record(("114", 600.0), ("122", 1200.0), ("209", 600.0))
    )

    midpoints = [segment["midpoint_minutes"] for segment in segments]

    # 0-10, 10-30 and 30-40 minutes: centres at 5, 20 and 35.
    assert midpoints == [
        pytest.approx(5.0),
        pytest.approx(20.0),
        pytest.approx(35.0),
    ]


def test_a_truncated_final_record_keeps_its_shorter_span_and_label():
    # The 210-minute run's shape: six full records then a truncated 114.
    full_records = [
        (name, RECORD_SECONDS) for name in ("114", "122", "209", "210", "231", "233")
    ]
    segments = record_segments(_per_record(*full_records, ("114", 1766.6)))

    assert [segment["record_name"] for segment in segments] == [
        "114",
        "122",
        "209",
        "210",
        "231",
        "233",
        "114",
    ]
    # Six interior transitions, none at the truncated end.
    assert len(transition_minutes(segments)) == 6
    assert segments[-1]["end_minutes"] == pytest.approx(
        (6 * RECORD_SECONDS + 1766.6) / 60
    )


def test_segments_load_from_a_result_json(tmp_path):
    path = tmp_path / "fp32_sustained_210min.json"
    path.write_text(
        json.dumps(
            {
                "streaming": {
                    "per_record": _per_record(
                        ("114", RECORD_SECONDS),
                        ("122", RECORD_SECONDS),
                    )
                }
            }
        )
    )

    segments = load_record_segments(path)

    assert [segment["record_name"] for segment in segments] == ["114", "122"]


def test_a_missing_or_malformed_result_json_degrades_to_none(tmp_path):
    missing = tmp_path / "absent.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"unexpected": "shape"}))
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"streaming": {"per_record": []}}))

    assert load_record_segments(missing) is None
    assert load_record_segments(malformed) is None
    assert load_record_segments(empty) is None


def test_the_result_json_is_found_beside_the_telemetry_npz(tmp_path):
    npz = tmp_path / "fp32_sustained_210min_raw.npz"
    npz.write_bytes(b"")
    result = tmp_path / "fp32_sustained_210min.json"
    result.write_text("{}")

    assert default_result_json(npz) == result


def test_no_companion_json_is_reported_as_none(tmp_path):
    npz = tmp_path / "fp32_sustained_210min_raw.npz"
    npz.write_bytes(b"")

    assert default_result_json(npz) is None
    assert default_result_json(tmp_path / "not_telemetry.npz") is None
