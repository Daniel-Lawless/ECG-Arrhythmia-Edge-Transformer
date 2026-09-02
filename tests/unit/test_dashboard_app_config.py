import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ecg_arrhythmia.dashboard import (
    display_mode,
    live_ecg_component,
    live_ecg_server,
    record_control,
    stream_service,
)


@pytest.mark.parametrize("public_url", [None, "http://127.0.0.1:9876/"])
def test_app_separates_server_binding_from_browser_url(monkeypatch, public_url):
    monkeypatch.setenv("ECG_DASHBOARD_HOST", "127.0.0.1")
    monkeypatch.setenv("ECG_DASHBOARD_PORT", "0")
    bind_host = "127.0.0.1" if public_url is None else "0.0.0.0"
    monkeypatch.setenv("ECG_LIVE_HTTP_HOST", bind_host)
    monkeypatch.setenv("ECG_LIVE_HTTP_PORT", "0")
    monkeypatch.delenv("ECG_LIVE_HTTP_PUBLIC_URL", raising=False)

    if public_url is not None:
        monkeypatch.setenv("ECG_LIVE_HTTP_PUBLIC_URL", public_url)

    service = SimpleNamespace(state=object(), host="127.0.0.1", port=0, start=Mock())
    endpoint = SimpleNamespace(host=bind_host, bound_port=15432, start=Mock())
    server_factory = Mock(return_value=endpoint)
    mount = Mock()
    monkeypatch.setattr(
        stream_service, "DashboardStreamService", Mock(return_value=service)
    )
    monkeypatch.setattr(live_ecg_server, "LiveEcgServer", server_factory)
    monkeypatch.setattr(live_ecg_component, "mount_live_dashboard", mount)
    monkeypatch.setattr(record_control, "render_record_control", Mock())
    monkeypatch.setattr(record_control, "control_endpoint", lambda: ("edge", 8767))
    monkeypatch.setattr(display_mode, "render_display_mode", lambda: "live")
    # Execute real app wiring without opening sockets or a browser session.
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(
            cache_resource=lambda function: function,
            set_page_config=Mock(),
            title=Mock(),
            caption=Mock(),
            markdown=Mock(),
            expander=lambda *args: nullcontext(),
        ),
    )
    app_path = (
        Path(__file__).resolve().parents[2] / "src/ecg_arrhythmia/dashboard/app.py"
    )
    runpy.run_path(str(app_path))

    server_factory.assert_called_once_with(service.state, host=bind_host, port=0)
    endpoint.start.assert_called_once()
    expected = (
        "http://127.0.0.1:15432" if public_url is None else public_url.rstrip("/")
    )
    assert mount.call_args.kwargs["endpoint_base"] == expected
