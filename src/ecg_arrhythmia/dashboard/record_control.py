import os

import streamlit as st

from ecg_arrhythmia.transport.control_client import (
    DEFAULT_PI_CONTROL_HOST,
    ControlClientError,
    start_record,
    stop_stream,
)
from ecg_arrhythmia.transport.control_config import DEFAULT_CONTROL_PORT
from ecg_arrhythmia.transport.control_protocol import (
    DEFAULT_DEMO_RECORD,
    DEMO_RECORDS,
    STATUS_OK,
    ControlProtocolError,
)

# Where the last command's outcome is parked between reruns.
_RESULT_KEY = "record_control_result"
_SELECT_KEY = "record_control_selection"


def control_endpoint() -> tuple[str, int]:
    """
    The Pi control address, from the environment or the link default.

    Matches how the rest of the dashboard resolves hosts and ports
    (ECG_DASHBOARD_*, ECG_LIVE_HTTP_*).
    """

    host = os.environ.get("ECG_PI_CONTROL_HOST", DEFAULT_PI_CONTROL_HOST)
    port = int(os.environ.get("ECG_PI_CONTROL_PORT", str(DEFAULT_CONTROL_PORT)))

    return host, port


def default_record_index(records=DEMO_RECORDS, default=DEFAULT_DEMO_RECORD) -> int:
    """Index of the default demo record, falling back to the first."""

    try:
        return records.index(default)
    except ValueError:
        return 0


def describe_response(response: dict) -> tuple[str, str]:
    """
    Turn an agent response into (severity, text) for the UI.

    Severity is "success" or "error"; the agent's own message is shown
    rather than a rewritten one, so the dashboard never claims an
    outcome the Pi did not report. Severity also selects the
    presentation: successes render as a compact inline badge in the
    control row, failures as a full-width alert beneath it.
    """

    message = response.get("message", "")

    if response.get("status") == STATUS_OK:
        return "success", f"Pi: {message}"

    return "error", f"Pi rejected the request: {message}"


def render_record_control() -> None:
    """Render the record selector and its Start/Stop buttons."""

    host, port = control_endpoint()

    # A horizontal container rather than columns: columns reserve a
    # share of the page width whether or not the widget fills it, so
    # any unused width became visible space between Start and Stop.
    # Here the children are laid out end to end at their own widths,
    # separated only by the container's small gap, and
    # vertical_alignment="bottom" puts the button bodies on the same
    # line as the select field rather than its label.
    #
    # Held as a named container rather than used only as a context
    # manager so the acknowledgement can be appended to the same row
    # further down, after the command has actually run.
    control_strip = st.container(
        horizontal=True,
        horizontal_alignment="left",
        vertical_alignment="bottom",
        gap="small",
    )

    with control_strip:
        # The buttons size to their labels by default, but a selectbox
        # defaults to width="stretch" and would otherwise expand to
        # fill the row. Its width parameter takes pixels or "stretch"
        # (not "content"), so an explicit width is the native way to
        # keep it compact: enough for a three-digit record and the
        # chevron, and for the label above to stay on one line.
        record = st.selectbox(
            "MIT-BIH demo record",
            DEMO_RECORDS,
            index=default_record_index(),
            key=_SELECT_KEY,
            width=150,
            help=(
                "Replayed on the Raspberry Pi through the full inference "
                "pipeline. Paced records are not offered: paced beats are "
                "outside the four AAMI classes this model was trained on."
            ),
        )
        start_clicked = st.button("Start stream")
        stop_clicked = st.button("Stop")

    # st.button returns True only on the rerun caused by the click, so
    # each command is sent exactly once however many reruns follow.
    if start_clicked:
        st.session_state[_RESULT_KEY] = _run_command(
            start_record,
            host=host,
            port=port,
            record=record,
        )
    elif stop_clicked:
        st.session_state[_RESULT_KEY] = _run_command(
            stop_stream,
            host=host,
            port=port,
        )

    result = st.session_state.get(_RESULT_KEY)

    if result is not None:
        severity, text = result

        if severity == "success":
            # A routine acknowledgement, sized to its own text and
            # appended to the control row itself. st.success would be
            # a full-width alert here: its width parameter takes only
            # pixels or "stretch", so it cannot shrink to content,
            # and a banner overstates a message this small.
            control_strip.badge(
                text,
                icon=":material/check_circle:",
                color="green",
            )
        else:
            # Failures keep the prominent full-width alert below the
            # controls: an unreachable Pi or a refused command should
            # not be reduced to a small inline note.
            st.error(text, icon=":material/error:")


def _run_command(command, host: str, port: int, record: str | None = None):
    """Call one control function and reduce it to (severity, text)."""

    try:
        if record is None:
            response = command(host=host, port=port)
        else:
            response = command(record, host=host, port=port)
    except ControlProtocolError as error:
        # Only reachable if the offered options and the agent's
        # allowlist ever diverge.
        return "error", f"Invalid request: {error}"
    except ControlClientError as error:
        return "error", str(error)

    return describe_response(response)
