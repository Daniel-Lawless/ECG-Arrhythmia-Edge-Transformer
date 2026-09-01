import math

import pytest

from ecg_arrhythmia.dashboard.presentation import (
    CLASS_DISPLAY,
    CLASS_ORDER,
    ECG_REFRESH_SECONDS,
    UNAVAILABLE,
    class_display,
    condition_text,
    connection_label,
    format_age_seconds,
    format_core_percent,
    format_frequency_mhz,
    format_milliseconds,
    format_temperature,
    latest_prediction,
    recent_beats,
    record_time_seconds,
    rhythm_from_prediction,
    stable_softmax,
    yes_no,
)
from ecg_arrhythmia.dashboard.state import (
    ReceivedPrediction,
    ReceivedRuntimeStatus,
)


def _prediction(
    target=2358,
    peaks=(2000, 2358),
    label="N",
    logits=(1.0, 2.0, 3.0, 4.0),
):
    return ReceivedPrediction(
        record_name="114",
        target_peak_index=target,
        peak_indices=tuple(peaks),
        logits=tuple(logits),
        predicted_class_index=0,
        predicted_label=label,
    )


# ---------------------------------------------------------------------
#                          Stable Softmax
# ---------------------------------------------------------------------


def test_softmax_scores_are_normalised_and_ordered():
    scores = stable_softmax([1.0, 2.0, 3.0, 4.0])

    assert all(score >= 0 for score in scores)
    assert sum(scores) == pytest.approx(1.0)
    # Argmax preserved: the largest logit keeps the largest score.
    assert scores.index(max(scores)) == 3
    assert scores[0] < scores[1] < scores[2] < scores[3]


def test_softmax_is_numerically_stable_for_huge_logits():
    scores = stable_softmax([1000.0, 1000.0, 1000.0, 1000.0])

    assert scores == pytest.approx((0.25, 0.25, 0.25, 0.25))

    dominated = stable_softmax([0.0, 0.0, 0.0, 1e6])

    assert all(math.isfinite(score) for score in dominated)
    assert dominated[3] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad_logits",
    [
        None,
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, float("nan"), 2.0, 3.0],
        [1.0, float("inf"), 2.0, 3.0],
        ["a", "b", "c", "d"],
    ],
)
def test_malformed_logits_yield_none_not_a_crash(bad_logits):
    assert stable_softmax(bad_logits) is None


# ---------------------------------------------------------------------
#                          Class Display
# ---------------------------------------------------------------------


def test_all_four_classes_have_distinct_display_metadata():
    assert set(CLASS_DISPLAY) == set(CLASS_ORDER) == {"N", "S", "V", "F"}

    colours = [display.colour for display in CLASS_DISPLAY.values()]
    names = [display.name for display in CLASS_DISPLAY.values()]

    assert len(set(colours)) == 4
    assert len(set(names)) == 4


def test_an_unknown_label_gets_safe_display_metadata():
    display = class_display("Q")

    assert display.label == "Q"
    assert display.name == "Unknown class"
    assert display.colour not in {d.colour for d in CLASS_DISPLAY.values()}


# ---------------------------------------------------------------------
#                        Latest / Recent Beats
# ---------------------------------------------------------------------


def test_latest_prediction_is_the_most_recently_received():
    first = _prediction(target=100)
    second = _prediction(target=200)

    assert latest_prediction(()) is None
    assert latest_prediction((first, second)).target_peak_index == 200


def test_recent_beats_selects_the_latest_without_mutating_the_source():
    source = [_prediction(target=index) for index in range(20)]

    beats = recent_beats(source, limit=12)

    # Oldest to newest, exactly the last twelve.
    assert [beat.target_peak_index for beat in beats] == list(range(8, 20))
    assert len(source) == 20
    assert source[0].target_peak_index == 0


# ---------------------------------------------------------------------
#                             Rhythm
# ---------------------------------------------------------------------


def test_rr_and_hr_derive_from_the_final_two_peaks():
    prediction = _prediction(peaks=(500, 1000, 1300))

    rhythm = rhythm_from_prediction(prediction, sampling_rate=360.0)

    assert rhythm.rr_seconds == pytest.approx(300 / 360)
    assert rhythm.hr_bpm == pytest.approx(60 / (300 / 360))
    assert rhythm.hr_bpm == pytest.approx(72.0)


@pytest.mark.parametrize(
    ("prediction", "rate"),
    [
        (None, 360.0),
        (_prediction(peaks=(1000,)), 360.0),
        (_prediction(peaks=(1000, 1300)), None),
        (_prediction(peaks=(1000, 1300)), 0.0),
        (_prediction(peaks=(1300, 1300)), 360.0),
        (_prediction(peaks=(1300, 1000)), 360.0),
    ],
)
def test_underivable_rhythm_is_unavailable_not_an_error(prediction, rate):
    rhythm = rhythm_from_prediction(prediction, rate)

    assert rhythm.rr_seconds is None
    assert rhythm.hr_bpm is None


# ---------------------------------------------------------------------
#                           Formatting
# ---------------------------------------------------------------------


def test_nullable_values_render_as_em_dash_never_zero():
    assert format_temperature(None) == UNAVAILABLE
    assert format_core_percent(None) == UNAVAILABLE
    assert format_frequency_mhz(None) == UNAVAILABLE
    assert format_milliseconds(None) == UNAVAILABLE
    assert format_age_seconds(None) == UNAVAILABLE


def test_present_values_format_consistently():
    assert format_temperature(48.7) == "48.7 °C"
    assert format_core_percent(3.5) == "3.5%"
    assert format_frequency_mhz(2400.0) == "2400 MHz"
    assert format_milliseconds(1.4) == "1.4 ms"
    assert format_age_seconds(0.05) == "0.05 s"
    assert format_age_seconds(12.0) == "12.00 s"


def test_record_time_requires_a_positive_sampling_rate():
    assert record_time_seconds(46217, 360.0) == pytest.approx(128.38, abs=0.01)
    assert record_time_seconds(46217, None) is None
    assert record_time_seconds(None, 360.0) is None
    assert record_time_seconds(46217, 0.0) is None


# ---------------------------------------------------------------------
#                        Status Presentation
# ---------------------------------------------------------------------


def test_connection_labels_cover_the_real_backend_states():
    assert connection_label("connected") == "Connected"
    assert connection_label("listening") == "Listening for Pi"
    assert connection_label("disconnected") == "Disconnected"
    # Unknown strings pass through rather than inventing states.
    assert connection_label("odd") == "odd"


def test_the_live_poll_cadence_matches_the_chunk_cadence():
    # The single client-side polling loop targets the 36-sample /
    # 360 Hz chunk cadence (100 ms). A presentation cadence only,
    # never a real-time claim.
    assert ECG_REFRESH_SECONDS == pytest.approx(36 / 360.0)


def test_display_modes_default_to_presentation_and_avoid_realtime_wording():
    from ecg_arrhythmia.dashboard.presentation import (
        DEFAULT_DISPLAY_MODE,
        DISPLAY_MODE_HELP,
        DISPLAY_MODES,
    )

    assert DISPLAY_MODES == ("Presentation", "Live")
    assert DEFAULT_DISPLAY_MODE == "Presentation"
    # Both modes run on the same real-time pipeline, so neither is
    # named "real-time" - that would imply the other one is not.
    assert "Real-time" not in DISPLAY_MODES
    assert "readability" in DISPLAY_MODE_HELP
    assert "newest" in DISPLAY_MODE_HELP


def test_the_mode_selects_only_the_dwell_duration():
    from ecg_arrhythmia.dashboard.presentation import (
        LIVE_HOLD_SECONDS,
        PREDICTION_PRESENTATION_SECONDS,
        hold_seconds_for_mode,
    )

    # Both modes run the same sequential FIFO; the hold is the entire
    # difference between them.
    assert hold_seconds_for_mode("Presentation") == PREDICTION_PRESENTATION_SECONDS
    assert hold_seconds_for_mode("Live") == LIVE_HOLD_SECONDS
    assert LIVE_HOLD_SECONDS == 0.1
    # Live is quicker but never instant: a zero hold would collapse
    # several predictions into one browser paint.
    assert 0 < LIVE_HOLD_SECONDS < PREDICTION_PRESENTATION_SECONDS
    # An unrecognised value falls back to the readable hold rather
    # than silently speeding the dashboard up.
    assert hold_seconds_for_mode("nonsense") == PREDICTION_PRESENTATION_SECONDS


def test_power_thermal_condition_uses_three_valued_semantics():
    assert condition_text(False) == "OK"
    assert condition_text(True) == "Warning"
    assert condition_text(None) == "Unknown"


def test_literal_throttling_is_presented_independently_of_the_aggregate():
    # Under-voltage only: aggregate warns, literal throttling stays No.
    status = ReceivedRuntimeStatus(
        record_name="114",
        latest_sample_index=100,
        temperature_c=48.7,
        process_cpu_percent=3.5,
        process_rss_mib=253.0,
        available_ram_mib=610.0,
        cpu_frequency_mhz=2400.0,
        cpu_governor="performance",
        under_voltage_active=True,
        frequency_capped_active=False,
        throttling_active=False,
        soft_temp_limit_active=False,
        runtime_condition_occurred=True,
        window_max_chunk_processing_ms=1.4,
        window_min_processing_headroom_ms=98.6,
    )

    assert condition_text(status.runtime_condition_active) == "Warning"
    assert yes_no(status.throttling_active) == "No"
    assert yes_no(status.runtime_condition_occurred) == "Yes"
    assert yes_no(None) == UNAVAILABLE
