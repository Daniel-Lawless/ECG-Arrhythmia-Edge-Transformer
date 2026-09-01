import plotly.graph_objects as go

from ecg_arrhythmia.dashboard.presentation import (
    CLASS_ORDER,
    class_display,
)

ECG_LINE_COLOUR = "#8A8A8A"
ECG_FIGURE_HEIGHT = 340
SCORE_FIGURE_HEIGHT = 260

# Fixed score scale: 0-100% with a little headroom so outside text
# labels are not clipped. The visual magnitude must never rescale
# between predictions.
SCORE_AXIS_RANGE = (0.0, 105.0)

_DIMMED_ALPHA = 0.45


def _hex_to_rgba(colour: str, alpha: float) -> str:
    red = int(colour[1:3], 16)
    green = int(colour[3:5], 16)
    blue = int(colour[5:7], 16)

    return f"rgba({red},{green},{blue},{alpha})"


def _waiting_annotation(figure: go.Figure, text: str) -> None:
    figure.add_annotation(
        text=text,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font={"size": 16, "color": "#8A8A8A"},
    )


# ---------------------------------------------------------------------
#                            ECG Figure
# ---------------------------------------------------------------------


def ecg_figure(
    sample_indices,
    samples,
    sampling_rate,
    visible_predictions,
) -> go.Figure:
    """
    The rolling ECG window with class-coloured prediction markers.

    x is record-relative time (sample_index / sampling_rate); the
    authoritative location remains the integer sample index, carried
    into hover text. Each marker sits at its prediction's
    target_peak_index with y equal to the waveform amplitude at that
    exact index; predictions whose target is not inside the current
    window are skipped gracefully. Without samples or a sampling rate
    a valid waiting figure is returned.
    """

    figure = go.Figure()
    figure.update_layout(
        height=ECG_FIGURE_HEIGHT,
        margin={"l": 55, "r": 15, "t": 35, "b": 45},
        xaxis_title="Time in record (s)",
        yaxis_title="ECG amplitude (mV)",
        legend={"orientation": "h", "y": 1.14, "x": 0.0},
        showlegend=True,
    )

    if not samples or sampling_rate is None or sampling_rate <= 0:
        _waiting_annotation(figure, "Waiting for ECG stream…")

        return figure

    times = [index / sampling_rate for index in sample_indices]

    figure.add_trace(
        go.Scattergl(
            x=times,
            y=list(samples),
            mode="lines",
            line={"color": ECG_LINE_COLOUR, "width": 1},
            name="ECG",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    amplitude_at = dict(zip(sample_indices, samples, strict=True))

    for label in CLASS_ORDER:
        display = class_display(label)
        marked = [
            prediction
            for prediction in visible_predictions
            if prediction.predicted_label == label
            and prediction.target_peak_index in amplitude_at
        ]

        if not marked:
            continue

        figure.add_trace(
            go.Scattergl(
                x=[
                    prediction.target_peak_index / sampling_rate
                    for prediction in marked
                ],
                y=[amplitude_at[prediction.target_peak_index] for prediction in marked],
                mode="markers",
                marker={
                    "color": display.colour,
                    "size": 9,
                    "line": {"color": "white", "width": 1},
                },
                name=f"{display.label} — {display.name}",
                customdata=[
                    [display.name, prediction.target_peak_index]
                    for prediction in marked
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "sample %{customdata[1]}<br>"
                    "%{x:.2f} s<extra></extra>"
                ),
            )
        )

    figure.update_xaxes(range=[times[0], times[-1]])

    return figure


# ---------------------------------------------------------------------
#                        Class-Score Figure
# ---------------------------------------------------------------------


def class_score_figure(scores, predicted_label) -> go.Figure:
    """
    Four horizontal bars of softmax-normalised class scores.

    Fixed 0-100% scale so magnitude never rescales between
    predictions; the argmax class keeps its full colour while the
    others are dimmed. scores=None yields a valid waiting figure.
    """

    figure = go.Figure()
    figure.update_layout(
        height=SCORE_FIGURE_HEIGHT,
        margin={"l": 30, "r": 20, "t": 15, "b": 35},
        xaxis={
            "range": list(SCORE_AXIS_RANGE),
            "tickvals": [0, 25, 50, 75, 100],
            "ticksuffix": "%",
        },
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )

    if scores is None:
        _waiting_annotation(figure, "Waiting for first prediction…")

        return figure

    percentages = [score * 100.0 for score in scores]
    colours = []

    for label in CLASS_ORDER:
        colour = class_display(label).colour

        if label == predicted_label:
            colours.append(colour)
        else:
            colours.append(_hex_to_rgba(colour, _DIMMED_ALPHA))

    figure.add_trace(
        go.Bar(
            x=percentages,
            y=list(CLASS_ORDER),
            orientation="h",
            marker={"color": colours},
            text=[f"{percentage:.1f}%" for percentage in percentages],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )

    return figure
