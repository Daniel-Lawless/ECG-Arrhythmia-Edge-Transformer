import os

import streamlit as st

from ecg_arrhythmia.dashboard import presentation
from ecg_arrhythmia.dashboard.display_mode import render_display_mode
from ecg_arrhythmia.dashboard.live_ecg_component import mount_live_dashboard
from ecg_arrhythmia.dashboard.live_ecg_server import (
    DEFAULT_LIVE_ECG_HOST,
    DEFAULT_LIVE_ECG_PORT,
    LiveEcgServer,
)
from ecg_arrhythmia.dashboard.record_control import (
    control_endpoint,
    render_record_control,
)
from ecg_arrhythmia.dashboard.state import DashboardState
from ecg_arrhythmia.dashboard.stream_service import DashboardStreamService
from ecg_arrhythmia.transport.protocol import SCHEMA_VERSION
from ecg_arrhythmia.transport.tcp_receiver import DEFAULT_PORT


@st.cache_resource
def stream_service() -> DashboardStreamService:
    """
    The process-wide stream service singleton.

    st.cache_resource guarantees one instance per Streamlit server
    process, so reruns and additional browser sessions never spawn a
    second receiver thread or fight over the TCP port.
    """

    host = os.environ.get("ECG_DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("ECG_DASHBOARD_PORT", str(DEFAULT_PORT)))

    service = DashboardStreamService(DashboardState(), host=host, port=port)
    service.start()

    return service


@st.cache_resource
def live_ecg_server() -> LiveEcgServer:
    """
    The process-wide local live-endpoint singleton.

    Reads the SAME DashboardState as the stream service - never a
    second Pi pipeline - and exists once per Streamlit process however
    many tabs poll it.
    """

    host = os.environ.get("ECG_LIVE_HTTP_HOST", DEFAULT_LIVE_ECG_HOST)
    port = int(os.environ.get("ECG_LIVE_HTTP_PORT", str(DEFAULT_LIVE_ECG_PORT)))

    server = LiveEcgServer(stream_service().state, host=host, port=port)
    server.start()

    return server


# ---------------------------------------------------------------------
#                        Static Page Shell
# ---------------------------------------------------------------------


st.set_page_config(
    page_title="Real-Time ECG Arrhythmia Edge Inference",
    layout="wide",
)

st.title("Real-Time ECG Arrhythmia Edge Inference")
st.caption(
    "Raspberry Pi 5 → ONNX Transformer → Live PC Dashboard · "
    "Research prototype — not for clinical use."
)

# The single dynamic surface: one persistent client-side component,
# one ~100 ms polling loop, every live panel updated together from one
# atomic snapshot payload.
_service = stream_service()
_endpoint = live_ecg_server()

# Static control strip: sends one command to the Pi agent per click and
# never participates in the live update path below.
render_record_control()

# Presentation-layer only: chooses which prediction the Classification
# and Model output panels show, never what the Pi computes or sends.
_display_mode = render_display_mode()

_control_host, _control_port = control_endpoint()

mount_live_dashboard(
    endpoint_base=f"http://{_endpoint.host}:{_endpoint.bound_port}",
    poll_ms=int(presentation.ECG_REFRESH_SECONDS * 1000),
    show_diagnostics=False,
    display_mode=_display_mode,
)

with st.expander("Deployment details (static)"):
    st.markdown(
        "\n".join(
            [
                "- Precision: FP32 (Section 5.5 deployment default)",
                "- Execution provider: CPUExecutionProvider",
                f"- Pi TCP transport listener: {_service.host}:{_service.port}",
                f"- Live dashboard endpoint: {_endpoint.host}:{_endpoint.bound_port}",
                f"- Pi control agent: {_control_host}:{_control_port}",
                f"- Protocol schema version: {SCHEMA_VERSION}",
                "- Live values (record, governor, telemetry, counters) "
                "appear in the live panel above.",
            ]
        )
    )
