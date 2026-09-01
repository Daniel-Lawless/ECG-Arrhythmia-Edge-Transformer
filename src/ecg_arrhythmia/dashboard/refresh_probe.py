import time

import streamlit as st

PROBE_REFRESH_SECONDS = 0.1
PROBE_LOG_EVERY_FRAMES = 20

st.set_page_config(page_title="Streamlit refresh probe")
st.title("Streamlit fragment refresh probe")
st.caption(
    "Development diagnostic — bare st.fragment(run_every=0.1) cadence "
    "with no Plotly, no streaming and no project state."
)


@st.fragment(run_every=PROBE_REFRESH_SECONDS)
def probe_fragment() -> None:
    now = time.perf_counter()
    record = st.session_state.setdefault(
        "_probe",
        {"previous": None, "intervals": [], "frames": 0, "latest": None},
    )

    record["frames"] += 1

    if record["previous"] is not None:
        interval = now - record["previous"]
        record["latest"] = interval
        record["intervals"].append(interval)

    record["previous"] = now

    st.metric("Frames", record["frames"])
    st.metric(
        "Latest interval",
        f"{record['latest']:.3f} s" if record["latest"] is not None else "n/a",
    )

    if len(record["intervals"]) >= PROBE_LOG_EVERY_FRAMES:
        intervals = record["intervals"]
        print(
            f"REFRESH_PROBE "
            f"interval_mean={sum(intervals) / len(intervals):.3f}s "
            f"interval_max={max(intervals):.3f}s "
            f"frames={record['frames']}",
            flush=True,
        )
        record["intervals"] = []


probe_fragment()
