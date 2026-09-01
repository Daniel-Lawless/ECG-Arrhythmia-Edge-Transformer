import pytest

from ecg_arrhythmia.dashboard.plots import (
    SCORE_AXIS_RANGE,
    class_score_figure,
    ecg_figure,
)
from ecg_arrhythmia.dashboard.presentation import CLASS_ORDER, class_display
from ecg_arrhythmia.dashboard.state import ReceivedPrediction

SAMPLING_RATE = 100.0


def _prediction(target, label="V"):
    return ReceivedPrediction(
        record_name="114",
        target_peak_index=target,
        peak_indices=(target - 50, target),
        logits=(1.0, 2.0, 3.0, 4.0),
        predicted_class_index=2,
        predicted_label=label,
    )


def _figure_data():
    sample_indices = tuple(range(1000, 1006))
    samples = (0.1, 0.2, 0.9, 0.3, -0.1, 0.0)

    return sample_indices, samples


# ---------------------------------------------------------------------
#                            ECG Figure
# ---------------------------------------------------------------------


def test_waveform_axes_derive_from_indices_rate_and_samples():
    sample_indices, samples = _figure_data()

    figure = ecg_figure(sample_indices, samples, SAMPLING_RATE, ())

    waveform = figure.data[0]

    assert waveform.x == pytest.approx(
        tuple(index / SAMPLING_RATE for index in sample_indices)
    )
    assert waveform.y == pytest.approx(samples)
    # WebGL trace: this figure is redrawn ~10x/s in the live fragment.
    assert waveform.type == "scattergl"
    assert figure.layout.xaxis.title.text == "Time in record (s)"
    assert figure.layout.yaxis.title.text == "ECG amplitude (mV)"


def test_markers_sit_on_the_waveform_at_their_target_index():
    sample_indices, samples = _figure_data()
    prediction = _prediction(1002, label="V")

    figure = ecg_figure(sample_indices, samples, SAMPLING_RATE, (prediction,))

    display = class_display("V")
    markers = next(trace for trace in figure.data if trace.name.startswith("V"))

    assert markers.x == pytest.approx((1002 / SAMPLING_RATE,))
    # y is the actual waveform amplitude at index 1002.
    assert markers.y == pytest.approx((0.9,))
    assert markers.type == "scattergl"
    assert markers.marker.color == display.colour
    # Hover carries the class name and authoritative sample index.
    assert markers.customdata[0][0] == display.name
    assert markers.customdata[0][1] == 1002


def test_a_prediction_outside_the_window_is_skipped_gracefully():
    sample_indices, samples = _figure_data()

    figure = ecg_figure(
        sample_indices,
        samples,
        SAMPLING_RATE,
        (_prediction(900, label="V"), _prediction(1003, label="N")),
    )

    trace_names = [trace.name for trace in figure.data[1:]]

    # Only the in-window N prediction produced a marker trace.
    assert len(trace_names) == 1
    assert trace_names[0].startswith("N")


def test_the_x_axis_advances_with_each_newer_snapshot():
    # Successive fragment reruns rebuild the figure from the newest
    # snapshot: the plotted span must track the advancing window with
    # no state carried over between calls.
    earlier = ecg_figure(
        tuple(range(1000, 1006)),
        (0.1,) * 6,
        SAMPLING_RATE,
        (),
    )
    later = ecg_figure(
        tuple(range(1010, 1016)),
        (0.2,) * 6,
        SAMPLING_RATE,
        (),
    )

    assert tuple(earlier.layout.xaxis.range) == (10.0, 10.05)
    assert tuple(later.layout.xaxis.range) == (10.1, 10.15)
    # The earlier figure is untouched by the later build.
    assert earlier.data[0].y == pytest.approx((0.1,) * 6)


def test_an_empty_snapshot_yields_a_waiting_figure_not_an_error():
    for indices, samples, rate in (
        ((), (), SAMPLING_RATE),
        ((), (), None),
        (_figure_data()[0], _figure_data()[1], None),
    ):
        figure = ecg_figure(indices, samples, rate, ())

        assert len(figure.data) == 0
        assert "Waiting" in figure.layout.annotations[0].text


# ---------------------------------------------------------------------
#                         Class-Score Figure
# ---------------------------------------------------------------------


def test_score_bars_use_percentages_on_a_fixed_scale():
    scores = (0.072, 0.024, 0.891, 0.013)

    figure = class_score_figure(scores, "V")

    bars = figure.data[0]

    assert bars.y == tuple(CLASS_ORDER)
    assert bars.x == pytest.approx((7.2, 2.4, 89.1, 1.3))
    assert bars.text == ("7.2%", "2.4%", "89.1%", "1.3%")
    # The scale is fixed: it must never rescale between predictions.
    assert tuple(figure.layout.xaxis.range) == SCORE_AXIS_RANGE


def test_the_predicted_class_keeps_full_colour_and_others_are_dimmed():
    figure = class_score_figure((0.072, 0.024, 0.891, 0.013), "V")

    colours = figure.data[0].marker.color
    predicted_position = CLASS_ORDER.index("V")

    assert colours[predicted_position] == class_display("V").colour

    for position, colour in enumerate(colours):
        if position != predicted_position:
            assert colour.startswith("rgba(")


def test_missing_scores_yield_a_waiting_figure_not_an_error():
    figure = class_score_figure(None, None)

    assert len(figure.data) == 0
    assert "Waiting" in figure.layout.annotations[0].text
    assert tuple(figure.layout.xaxis.range) == SCORE_AXIS_RANGE
