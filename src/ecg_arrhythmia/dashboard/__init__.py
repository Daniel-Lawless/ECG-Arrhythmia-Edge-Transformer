"""
PC-side live dashboard state (Section 6.2).

Turns decoded transport messages into a thread-safe, bounded,
dashboard-ready live state: `state` holds the rolling ECG window,
recent predictions, counters and freshness behind immutable snapshots;
`stream_service` runs the Section 6.1 receiver on a background thread
and applies its messages to the state.

`state` and `stream_service` are standard-library only, so the
receiving side runs on a bare Windows Python. The Section 6.3 UI
layer on top of them - `presentation` (pure helpers), `plots`
(Plotly figures) and `app` (the Streamlit page) - needs the
`dashboard` extra (Streamlit + Plotly), which is PC-only: the
Raspberry Pi never installs it. This package's __init__ imports
nothing, so importing the lightweight modules never pulls UI
dependencies. The app owns exactly ONE stream service via
Streamlit's process-level resource cache; UI reruns must never
construct another.
"""
