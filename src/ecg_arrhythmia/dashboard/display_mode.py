import streamlit as st

from ecg_arrhythmia.dashboard.presentation import (
    DEFAULT_DISPLAY_MODE,
    DISPLAY_MODE_HELP,
    DISPLAY_MODES,
)

_MODE_KEY = "prediction_display_mode"


def selected_display_mode() -> str:
    """
    The current mode, defaulting before the widget has been rendered.

    Reading through session state means the value survives the reruns
    the record controls trigger.
    """

    return st.session_state.get(_MODE_KEY, DEFAULT_DISPLAY_MODE)


def render_display_mode() -> str:
    """Render the compact mode selector and return the chosen mode."""

    # Matches the record control strip: one horizontal container,
    # content-width widget, bottom-aligned.
    with st.container(
        horizontal=True,
        horizontal_alignment="left",
        vertical_alignment="bottom",
        gap="small",
    ):
        mode = st.segmented_control(
            "Display mode",
            DISPLAY_MODES,
            default=DEFAULT_DISPLAY_MODE,
            # required=True guarantees a mode is always selected, so
            # the caller never has to handle None.
            required=True,
            key=_MODE_KEY,
            help=DISPLAY_MODE_HELP,
        )

    return mode or DEFAULT_DISPLAY_MODE
