from ecg_arrhythmia.dashboard.live_ecg_server import RECENT_BEAT_LIMIT
from ecg_arrhythmia.dashboard.plots import ECG_FIGURE_HEIGHT, ECG_LINE_COLOUR
from ecg_arrhythmia.dashboard.presentation import (
    CLASS_DISPLAY,
    CLASS_ORDER,
    DEFAULT_DISPLAY_MODE,
    connection_colour,
    connection_label,
    hold_seconds_for_mode,
)

_CONNECTION_STATUSES = ("connected", "listening", "disconnected")

# Bound on the client-side history of already-presented beat markers.
# At the fixed presentation pace this comfortably covers every beat
# that could still sit inside the 10-second ECG window even when the
# queue lags well behind the live signal; markers outside the window
# are simply not drawn, and the array never grows past this.
MARKER_HISTORY_LIMIT = 64

# Cosmetic viewport smoothing: on ordinary contiguous updates the
# x-range glides linearly to each genuine new window over slightly
# less than the ~100 ms chunk cadence, so one movement finishes just
# before the next real chunk arrives instead of movements overlapping.
# ONLY the visible range is animated - never ECG samples, marker
# positions or anything timing-derived; the plotted data stays
# authoritative and entirely genuine.
ECG_VIEWPORT_ANIMATION_MS = 85

# Viewport jumps larger than this snap straight to the genuine range
# rather than crawling through stale time (tab stalls, long delays,
# reconnects): five nominal 100 ms chunk periods.
VIEWPORT_SNAP_SECONDS = 0.5


def ecg_update_decision(previous: dict | None, payload: dict) -> str:
    """
    Append-vs-rebuild choice for the incremental ECG path.

    Pure mirror of the JS updateDecision() (which is authoritative in
    the browser); kept in sync so the scenarios are testable without a
    browser. `previous` holds the last rendered chart state
    ({record_name, sampling_rate, latest_sample_index}) or None before
    the first render. Returns:

        "rebuild"  full Plotly.react (first render, waiting state,
                   record/rate change, regression, unbridgeable gap)
        "append"   contiguous advance: extend with the new samples only
        "hold"     nothing new: no waveform change needed
    """

    ecg = payload["ecg"]

    if not ecg["samples"] or not payload["sampling_rate"]:
        return "rebuild"

    if previous is None:
        return "rebuild"

    if payload["record_name"] != previous["record_name"]:
        return "rebuild"

    if payload["sampling_rate"] != previous["sampling_rate"]:
        return "rebuild"

    latest = ecg["latest_sample_index"]
    previous_latest = previous["latest_sample_index"]

    if latest is None or previous_latest is None:
        return "rebuild"

    if latest < previous_latest:
        return "rebuild"

    if latest == previous_latest:
        return "hold"

    if previous_latest + 1 < ecg["start_index"]:
        return "rebuild"

    return "append"


# Hover tooltips for the two model metrics; clarification stays in
# tooltips rather than visible prose.
MODEL_LATENCY_TOOLTIP = (
    "Mean model-stage latency from the most recent live Pi inference measurement."
)
MODEL_THROUGHPUT_TOOLTIP = (
    "Active model-stage sequence capacity derived from timed inference "
    "work; not ECG prediction rate."
)

_HTML = f"""
<div id="live-dash">
  <div id="ld-status-row">
    <div id="ld-conn">
      <span id="ld-conn-dot">●</span> <b id="ld-conn-text">Listening for Pi</b>
    </div>
    <div class="ld-metric">
      <div class="ld-metric-label">Record</div>
      <div class="ld-metric-value" id="ld-record">—</div>
    </div>
    <div class="ld-metric">
      <div class="ld-metric-label">Stream freshness</div>
      <div class="ld-metric-value" id="ld-freshness">—</div>
    </div>
    <div class="ld-metric">
      <div class="ld-metric-label">Discontinuities</div>
      <div class="ld-metric-value" id="ld-gaps">0</div>
    </div>
  </div>

  <div id="ecg-chart" style="width:100%; height:{ECG_FIGURE_HEIGHT}px;"></div>
  <div id="ecg-diag"></div>

  <div id="ld-mid-row">
    <div>
      <div class="ld-panel-title">Classification</div>
      <div id="ld-class-label">—</div>
      <div id="ld-class-name" class="ld-caption">
        Waiting for first prediction…
      </div>
      <div id="ld-class-sample" class="ld-small"></div>
      <div id="ld-class-time" class="ld-small"></div>
      <div id="ld-class-queue" class="ld-caption"></div>
      <div id="ld-class-oob" class="ld-caption"></div>
    </div>
    <div>
      <div class="ld-panel-title">Model output</div>
      <div class="ld-caption">
        Softmax-normalised class scores — not calibrated probabilities.
      </div>
      <div id="ld-bars"></div>
    </div>
  </div>

  <div id="ld-bottom-row">
    <div>
      <div class="ld-panel-title">Rhythm</div>
      <div class="ld-metric">
        <div class="ld-metric-label">Estimated HR</div>
        <div class="ld-metric-value" id="ld-hr">—</div>
      </div>
      <div class="ld-metric">
        <div class="ld-metric-label">Latest RR</div>
        <div class="ld-metric-value" id="ld-rr">—</div>
      </div>
    </div>
    <div>
      <div class="ld-panel-title">Edge runtime</div>
      <div id="ld-runtime-grid">
        <div class="ld-metric">
          <div class="ld-metric-label">Temp</div>
          <div class="ld-metric-value" id="ld-rt-temp">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label">CPU (% of one core)</div>
          <div class="ld-metric-value" id="ld-rt-cpu">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label">RSS</div>
          <div class="ld-metric-value" id="ld-rt-rss">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label">Available RAM</div>
          <div class="ld-metric-value" id="ld-rt-ram">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label"
            title="{MODEL_LATENCY_TOOLTIP}">Model latency</div>
          <div class="ld-metric-value" id="ld-rt-model-latency">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label">CPU clock</div>
          <div class="ld-metric-value" id="ld-rt-clock">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label">Chunk processing</div>
          <div class="ld-metric-value" id="ld-rt-proc">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label">Processing headroom</div>
          <div class="ld-metric-value" id="ld-rt-headroom">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label">Telemetry age</div>
          <div class="ld-metric-value" id="ld-rt-age">—</div>
        </div>
        <div class="ld-metric">
          <div class="ld-metric-label"
            title="{MODEL_THROUGHPUT_TOOLTIP}">Model throughput</div>
          <div class="ld-metric-value" id="ld-rt-model-throughput">—</div>
        </div>
      </div>
      <div id="ld-rt-statusline" class="ld-small">
        Waiting for runtime telemetry…
      </div>
      <div id="ld-rt-historical" class="ld-caption"></div>
    </div>
  </div>

  <div id="ld-beats">
    <div class="ld-panel-title">Recent beats</div>
    <div id="ld-beats-strip">No predictions received yet.</div>
    <div class="ld-caption">Oldest to newest, left to right.</div>
  </div>
</div>
"""

_CSS = """
#live-dash { width: 100%; }
#ld-status-row {
  display: flex; gap: 2.5rem; align-items: center;
  margin-bottom: 0.5rem; flex-wrap: wrap;
}
#ld-conn { font-size: 1.15rem; }
#live-dash .ld-metric { min-width: 6rem; }
#live-dash .ld-metric-label { font-size: 0.8rem; color: #8a8a8a; }
#live-dash .ld-metric-value {
  font-size: 1.35rem; font-weight: 600;
  font-variant-numeric: tabular-nums;
}
#live-dash .ld-panel-title {
  font-size: 1.25rem; font-weight: 700; margin-bottom: 0.35rem;
}
#live-dash .ld-caption {
  font-size: 0.8rem; color: #8a8a8a; margin: 0.25rem 0;
}
#live-dash .ld-small { font-size: 0.9rem; margin-top: 0.15rem; }
#ld-mid-row {
  display: grid; grid-template-columns: 1fr 2fr; gap: 2rem;
  margin-top: 0.75rem;
}
#ld-bottom-row {
  display: grid; grid-template-columns: 1fr 3fr; gap: 2rem;
  margin-top: 1rem;
}
#ld-class-label { font-size: 3.6rem; font-weight: 700; line-height: 1.1; }
#live-dash .ld-bar-row {
  display: flex; align-items: center; gap: 0.6rem; margin: 0.35rem 0;
}
#live-dash .ld-bar-label { width: 1.2rem; font-weight: 700; }
#live-dash .ld-bar-track {
  flex: 1; height: 1.1rem; background: rgba(128,128,128,0.15);
  border-radius: 3px; overflow: hidden;
}
#live-dash .ld-bar-fill { height: 100%; width: 0%; }
#live-dash .ld-bar-pct {
  width: 3.6rem; text-align: right; font-variant-numeric: tabular-nums;
}
#ld-runtime-grid {
  display: grid; grid-template-columns: repeat(5, 1fr);
  gap: 0.6rem 1.5rem; margin-bottom: 0.4rem;
}
#ld-beats { margin-top: 1rem; }
#live-dash .ld-beat {
  font-size: 1.4rem; font-weight: 700; margin-right: 0.9rem;
}
#ecg-diag {
  font-size: 11px; color: #8a8a8a; font-family: monospace;
  padding: 2px 0 6px 0;
}
"""

_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const CONFIG = data;

  // A Streamlit rerun (record controls, display-mode switch)
  // re-executes this module against the same parentElement. Streamlit
  // calls the returned cleanup on unmount, but a re-execution without
  // an unmount would leave the previous instance's polling loop and
  // presentation state machine running: two loops would then fetch,
  // queue and render into the SAME panels, fighting over which
  // prediction is current. Tearing down whatever was mounted here
  // before keeps exactly one instance per element.
  if (typeof parentElement.__liveDashboardTeardown === "function") {
    parentElement.__liveDashboardTeardown();
  }

  const $ = (selector) => parentElement.querySelector(selector);

  const chart = $("#ecg-chart");
  const diag = $("#ecg-diag");
  const el = {
    connDot: $("#ld-conn-dot"),
    connText: $("#ld-conn-text"),
    record: $("#ld-record"),
    freshness: $("#ld-freshness"),
    gaps: $("#ld-gaps"),
    classLabel: $("#ld-class-label"),
    className: $("#ld-class-name"),
    classSample: $("#ld-class-sample"),
    classTime: $("#ld-class-time"),
    classQueue: $("#ld-class-queue"),
    classOob: $("#ld-class-oob"),
    bars: $("#ld-bars"),
    hr: $("#ld-hr"),
    rr: $("#ld-rr"),
    rtTemp: $("#ld-rt-temp"),
    rtCpu: $("#ld-rt-cpu"),
    rtRss: $("#ld-rt-rss"),
    rtRam: $("#ld-rt-ram"),
    rtClock: $("#ld-rt-clock"),
    rtProc: $("#ld-rt-proc"),
    rtHeadroom: $("#ld-rt-headroom"),
    rtAge: $("#ld-rt-age"),
    rtModelLatency: $("#ld-rt-model-latency"),
    rtModelThroughput: $("#ld-rt-model-throughput"),
    rtStatusLine: $("#ld-rt-statusline"),
    rtHistorical: $("#ld-rt-historical"),
    beatsStrip: $("#ld-beats-strip"),
  };

  const UNAVAILABLE = "\\u2014";
  const UNKNOWN_DISPLAY = {name: "Unknown class", colour: "#9A9A9A"};

  let updating = false;
  let stopped = false;
  let intervalHandle = null;
  let updateCount = 0;
  let previousTick = null;
  const intervals = [];

  // Presentation and chart state, held on the ELEMENT so it survives
  // re-execution of this module.
  //
  // A Streamlit rerun (changing the display mode, or using the record
  // controls) runs this file again against the same parentElement. If
  // this state lived in the closure it would start empty every time,
  // presentationSeeded would be false, and the mid-stream seeding
  // path would rebuild a dozen markers and beats out of recent_beats
  // in a single update - which is what appeared as a batch of
  // predictions all landing at once. Persisting it means a mode
  // change alters only the display policy: the presentation engine
  // keeps running with the same queue, the same seen-event set and
  // the same history.
  //
  // Genuine stream changes are unaffected: resetPresentation mutates
  // this same object, so a record change or a sample-index regression
  // still clears everything exactly as before.
  //
  // currentPresented remains the SINGLE source of truth for the event
  // being shown: it drives the Classification panel, the Model output
  // bars, AND the active ECG marker, so their correspondence is
  // unambiguous by construction. presentedMarkerHistory holds the
  // same event objects AFTER their presentation turn ends, giving the
  // ECG its faded short-term history; queue-waiting predictions never
  // appear in either, so an ECG circle always means "a prediction the
  // viewer has been shown".
  const session = parentElement.__liveDashboardSession || (
    parentElement.__liveDashboardSession = {
      seenPredictionIds: new Set(),
      presentationQueue: [],
      presentedMarkerHistory: [],
      presentedBeats: [],
      currentPresented: null,
      presentedAt: null,
      presentationSeeded: false,
      presentationRecord: null,
      presentationLatestSample: null,
      // Incremental-chart state: null until the first full render.
      chartState: null,
    }
  );

  // Aliases for the containers that are only ever mutated in place -
  // they share the persisted instance. Every value that gets
  // REASSIGNED is written through `session` instead, so the next
  // execution of this module observes it.
  const seenPredictionIds = session.seenPredictionIds;
  const presentationQueue = session.presentationQueue;
  const presentedMarkerHistory = session.presentedMarkerHistory;

  function fmt(value, digits, suffix) {
    if (value === null || value === undefined) return UNAVAILABLE;
    return value.toFixed(digits) + suffix;
  }

  function yesNo(flag) {
    if (flag === null || flag === undefined) return UNAVAILABLE;
    return flag ? "Yes" : "No";
  }

  // Model-output bars: built once per instance, then updated in place
  // on every advance. The container is cleared first because this
  // module re-executes against the SAME persisted DOM on a Streamlit
  // rerun (record controls, display-mode switch); appending without
  // clearing produced a second, third, ... N/S/V/F block each time.
  // Clearing makes the build idempotent however often it runs.
  const barRefs = {};

  el.bars.replaceChildren();

  for (const label of CONFIG.classOrder) {
    const display = CONFIG.classDisplay[label];
    const row = document.createElement("div");
    row.className = "ld-bar-row";
    const name = document.createElement("span");
    name.className = "ld-bar-label";
    name.textContent = label;
    name.style.color = display.colour;
    const track = document.createElement("div");
    track.className = "ld-bar-track";
    const fill = document.createElement("div");
    fill.className = "ld-bar-fill";
    fill.style.background = display.colour;
    const pct = document.createElement("span");
    pct.className = "ld-bar-pct";
    pct.textContent = UNAVAILABLE;
    track.appendChild(fill);
    row.appendChild(name);
    row.appendChild(track);
    row.appendChild(pct);
    el.bars.appendChild(row);
    barRefs[label] = {fill: fill, pct: pct};
  }

  const LAYOUT_BASE = {
    height: CONFIG.chartHeight,
    margin: {l: 55, r: 15, t: 35, b: 45},
    xaxis: {
      title: {text: "Time in record (s)"},
      showgrid: false,
      // Deterministic integer-second ticks: they enter/leave with the
      // moving window instead of alternating odd/even label sets.
      tickmode: "linear",
      tick0: 0,
      dtick: 1,
    },
    yaxis: {title: {text: "ECG amplitude (mV)"}, showgrid: false},
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#8a8a8a"},
    // No legend: with a single presented-beat marker whose class is
    // already explicit in the Classification panel, a legend would
    // only flicker as classes change.
    showlegend: false,
  };
  const PLOT_CONFIG = {displayModeBar: false, responsive: true};

  function renderWaitingChart(message) {
    const layout = Object.assign({}, LAYOUT_BASE, {
      annotations: [{
        text: message, showarrow: false, xref: "paper", yref: "paper",
        x: 0.5, y: 0.5, font: {size: 16, color: "#8a8a8a"},
      }],
    });
    cancelViewportAnimation();
    Plotly.react(chart, [], layout, PLOT_CONFIG);
    session.chartState = null;
    el.classOob.textContent = "";
  }

  // Mirrored by ecg_update_decision() in Python for tests; this JS
  // implementation is authoritative in the browser.
  function updateDecision(previous, payload) {
    const ecg = payload.ecg;

    if (!ecg.samples || ecg.samples.length === 0 || !payload.sampling_rate) {
      return "rebuild";
    }
    if (previous === null) return "rebuild";
    if (payload.record_name !== previous.recordName) return "rebuild";
    if (payload.sampling_rate !== previous.samplingRate) return "rebuild";

    const latest = ecg.latest_sample_index;
    const previousLatest = previous.latestSampleIndex;

    if (latest === null || previousLatest === null) return "rebuild";
    if (latest < previousLatest) return "rebuild";
    if (latest === previousLatest) return "hold";
    if (previousLatest + 1 < ecg.start_index) return "rebuild";

    return "append";
  }

  // One presented event -> one marker point, recomputed from the
  // absolute target_peak_index against the LIVE window on every
  // update, so markers stay locked to their beats while the waveform
  // scrolls; no screen coordinate is ever stored. A beat outside the
  // window yields null and is simply not drawn - never clamped, moved
  // to an edge, or placed on another beat.
  function markerPoint(event, payload) {
    const offset = event.target_peak_index - payload.ecg.start_index;

    if (offset < 0 || offset >= payload.ecg.samples.length) {
      return null;
    }

    const display = CONFIG.classDisplay[event.predicted_label]
      || UNKNOWN_DISPLAY;

    return {
      x: event.target_peak_index / payload.sampling_rate,
      y: payload.ecg.samples[offset],  // exact ECG amplitude
      colour: display.colour,
      custom: [display.name, event.target_peak_index],
    };
  }

  // Marker data for the two presented-marker traces. Historical =
  // events whose presentation turn has ended (faded, per-point class
  // colours); current = the event driving Classification and Model
  // output right now. Both derive ONLY from presentation state -
  // never from payload predictions the viewer has not been shown.
  function presentedMarkerData(payload) {
    const historical = {x: [], y: [], colours: [], custom: []};

    for (const event of presentedMarkerHistory) {
      const point = markerPoint(event, payload);
      if (point === null) continue;  // scrolled out: skip safely
      historical.x.push(point.x);
      historical.y.push(point.y);
      historical.colours.push(point.colour);
      historical.custom.push(point.custom);
    }

    const current = {x: [], y: [], colours: [], custom: [],
                     outOfWindow: false};

    if (session.currentPresented) {
      const point = markerPoint(session.currentPresented, payload);

      if (point === null) {
        current.outOfWindow = true;
      } else {
        current.x.push(point.x);
        current.y.push(point.y);
        current.colours.push(point.colour);
        current.custom.push(point.custom);
      }
    }

    return {historical: historical, current: current};
  }

  function updateOutOfWindowNote(current) {
    // Subtle, and only while the presented beat is genuinely outside
    // the visible window.
    el.classOob.textContent = current.outOfWindow
      ? "Beat no longer visible in ECG window"
      : "";
  }

  function restylePresentedMarkers(payload) {
    const markers = presentedMarkerData(payload);
    Plotly.restyle(chart, {
      x: [markers.historical.x, markers.current.x],
      y: [markers.historical.y, markers.current.y],
      customdata: [markers.historical.custom, markers.current.custom],
      "marker.color": [markers.historical.colours, markers.current.colours],
    }, [1, 2]);
    updateOutOfWindowNote(markers.current);
  }

  function xRangeFor(payload, firstTime) {
    const latestTime =
      payload.ecg.latest_sample_index / payload.sampling_rate;

    return [
      Math.max(firstTime, latestTime - CONFIG.windowSeconds),
      latestTime,
    ];
  }

  // Cosmetic viewport smoothing: on ordinary contiguous updates the
  // visible x-range glides linearly to the genuine new window via a
  // requestAnimationFrame loop of plain relayout steps. Plotly's own
  // transition engine is deliberately NOT used: its smooth
  // transitions are documented for SVG scatter only (these traces
  // are scattergl) and its frame queue is built for discrete
  // triggered sequences, not a 10 Hz replace-while-extending stream.
  // Data stays authoritative: only the viewport moves, never samples,
  // marker coordinates, or anything derived from them.
  let viewportAnimation = null;

  function cancelViewportAnimation() {
    if (viewportAnimation !== null) {
      cancelAnimationFrame(viewportAnimation);
      viewportAnimation = null;
    }
  }

  function animateViewportTo(targetRange) {
    // Newest genuine target always wins: any in-flight movement is
    // replaced immediately, so obsolete animations can never queue.
    cancelViewportAnimation();

    const currentRange = chart.layout && chart.layout.xaxis
      ? chart.layout.xaxis.range : null;

    if (!currentRange) {
      Plotly.relayout(chart, {"xaxis.range": targetRange});
      return;
    }

    const fromStart = currentRange[0];
    const fromEnd = currentRange[1];

    // Large/stale jumps snap straight to the genuine range - never a
    // slow catch-up crawl through stale time.
    if (Math.abs(targetRange[1] - fromEnd) > CONFIG.viewportSnapSeconds) {
      Plotly.relayout(chart, {"xaxis.range": targetRange});
      return;
    }

    const startedAt = performance.now();

    function step(now) {
      if (stopped) {
        viewportAnimation = null;
        return;
      }

      // Linear: an ECG monitor should sweep at constant speed.
      const progress = Math.min(
        (now - startedAt) / CONFIG.viewportAnimationMs, 1);
      Plotly.relayout(chart, {"xaxis.range": [
        fromStart + (targetRange[0] - fromStart) * progress,
        fromEnd + (targetRange[1] - fromEnd) * progress,
      ]});

      if (progress < 1) {
        viewportAnimation = requestAnimationFrame(step);
      } else {
        viewportAnimation = null;
      }
    }

    viewportAnimation = requestAnimationFrame(step);
  }

  function rebuildChart(payload) {
    const rate = payload.sampling_rate;
    const start = payload.ecg.start_index;
    const samples = payload.ecg.samples;
    const times = new Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      times[i] = (start + i) / rate;
    }

    const traces = [{
      x: times,
      y: samples.slice(),
      type: "scattergl",
      mode: "lines",
      line: {color: CONFIG.ecgLineColour, width: 1},
      name: "ECG",
      hoverinfo: "skip",
      showlegend: false,
    }];

    // Fixed trace layout: 0 = waveform, 1 = historical presented
    // markers, 2 = the current presented marker. Both marker traces
    // exist even when empty so indices are stable and no trace is
    // ever created/deleted at the poll cadence.
    const markers = presentedMarkerData(payload);

    traces.push({
      x: markers.historical.x,
      y: markers.historical.y,
      type: "scattergl",
      mode: "markers",
      // Visually secondary: same class colours (per point), faded so
      // the active (full-opacity, white-outlined) beat stays dominant.
      marker: {color: markers.historical.colours, size: 15, opacity: 0.55},
      name: "Presented beats",
      showlegend: false,
      customdata: markers.historical.custom,
      hovertemplate: "%{customdata[0]}<br>sample %{customdata[1]}" +
                     "<br>%{x:.2f} s<extra></extra>",
    });

    traces.push({
      x: markers.current.x,
      y: markers.current.y,
      type: "scattergl",
      mode: "markers",
      marker: {color: markers.current.colours, size: 15, opacity: 1,
               line: {color: "white", width: 1}},
      name: "Presented beat",
      showlegend: false,
      customdata: markers.current.custom,
      hovertemplate: "%{customdata[0]}<br>sample %{customdata[1]}" +
                     "<br>%{x:.2f} s<extra></extra>",
    });

    const layout = Object.assign({}, LAYOUT_BASE, {
      xaxis: Object.assign({}, LAYOUT_BASE.xaxis,
                           {range: xRangeFor(payload, times[0])}),
    });

    // Recovery/reset renders establish the correct range immediately:
    // no viewport animation, and no stale pending step may fire.
    cancelViewportAnimation();
    Plotly.react(chart, traces, layout, PLOT_CONFIG);
    updateOutOfWindowNote(markers.current);
  }

  function appendChart(payload, previousLatest) {
    const rate = payload.sampling_rate;
    const start = payload.ecg.start_index;
    const samples = payload.ecg.samples;

    // Only the genuinely new samples: previousLatest+1 .. latest.
    const deltaOffset = previousLatest + 1 - start;
    const deltaY = samples.slice(deltaOffset);
    const deltaX = new Array(deltaY.length);
    for (let i = 0; i < deltaY.length; i++) {
      deltaX[i] = (previousLatest + 1 + i) / rate;
    }

    // Window capacity derived from the live configuration, never a
    // hard-coded 3600.
    const capacity = Math.round(CONFIG.windowSeconds * rate);

    Plotly.extendTraces(chart, {x: [deltaX], y: [deltaY]}, [0], capacity);

    restylePresentedMarkers(payload);

    const firstTime = (start) / rate;
    animateViewportTo(xRangeFor(payload, firstTime));
  }

  function renderEcg(payload) {
    const decision = updateDecision(session.chartState, payload);

    if (decision === "rebuild") {
      if (!payload.ecg.samples || payload.ecg.samples.length === 0
          || !payload.sampling_rate) {
        renderWaitingChart("Waiting for ECG stream\\u2026");
        return;
      }
      rebuildChart(payload);
    } else if (decision === "append") {
      appendChart(payload, session.chartState.latestSampleIndex);
    } else {
      // "hold": no new samples, but the presentation queue may have
      // advanced on an unchanged window - refresh the markers only.
      restylePresentedMarkers(payload);
    }

    session.chartState = {
      recordName: payload.record_name,
      samplingRate: payload.sampling_rate,
      latestSampleIndex: payload.ecg.latest_sample_index,
    };
  }

  function renderStatus(payload) {
    const display = CONFIG.connectionDisplay[payload.connection_status]
      || {label: payload.connection_status, colour: "#9A9A9A"};
    el.connText.textContent = display.label;
    el.connDot.style.color = display.colour;
    el.record.textContent = payload.record_name || UNAVAILABLE;
    el.freshness.textContent = fmt(payload.stream_age_seconds, 2, " s");
    el.gaps.textContent = String(payload.discontinuities);
  }

  // ------------------- Sequential presentation -------------------

  function resetPresentation() {
    seenPredictionIds.clear();
    presentationQueue.length = 0;
    presentedMarkerHistory.length = 0;
    session.presentedBeats = [];
    session.currentPresented = null;
    session.presentedAt = null;
    session.presentationSeeded = false;
  }

  function renderPresented(presented) {
    if (!presented) {
      el.classLabel.textContent = UNAVAILABLE;
      el.classLabel.style.color = "";
      el.className.textContent = "Waiting for first prediction\\u2026";
      el.classSample.textContent = "";
      el.classTime.textContent = "";

      for (const label of CONFIG.classOrder) {
        barRefs[label].fill.style.width = "0%";
        barRefs[label].fill.style.opacity = "1";
        barRefs[label].pct.textContent = UNAVAILABLE;
      }

      return;
    }

    const display = CONFIG.classDisplay[presented.predicted_label]
      || UNKNOWN_DISPLAY;
    el.classLabel.textContent = presented.predicted_label;
    el.classLabel.style.color = display.colour;
    el.className.textContent = display.name;
    el.classSample.textContent =
      "Sample " + presented.target_peak_index.toLocaleString();
    el.classTime.textContent =
      fmt(presented.time_seconds, 2, " s into record");

    const scores = presented.scores;

    for (let i = 0; i < CONFIG.classOrder.length; i++) {
      const label = CONFIG.classOrder[i];
      const refs = barRefs[label];

      if (!scores) {
        refs.fill.style.width = "0%";
        refs.fill.style.opacity = "1";
        refs.pct.textContent = UNAVAILABLE;
        continue;
      }

      const pctValue = scores[i] * 100.0;
      refs.fill.style.width = pctValue.toFixed(1) + "%";
      refs.fill.style.opacity =
        (label === presented.predicted_label) ? "1" : "0.45";
      refs.pct.textContent = pctValue.toFixed(1) + "%";
    }
  }

  function renderBeatsStrip() {
    el.beatsStrip.textContent = "";

    if (session.presentedBeats.length === 0) {
      el.beatsStrip.textContent = "No predictions received yet.";
      return;
    }

    for (const beat of session.presentedBeats) {
      const display = CONFIG.classDisplay[beat.predicted_label]
        || UNKNOWN_DISPLAY;
      const span = document.createElement("span");
      span.className = "ld-beat";
      span.style.color = display.colour;
      span.textContent = beat.predicted_label;
      el.beatsStrip.appendChild(span);
    }
  }

  function presentNext(now) {
    // The outgoing event becomes historical in the SAME transition
    // that advances the panels - one timer, one source of truth.
    if (session.currentPresented !== null) {
      presentedMarkerHistory.push(session.currentPresented);

      while (presentedMarkerHistory.length > CONFIG.markerHistoryLength) {
        presentedMarkerHistory.shift();
      }
    }

    // FIFO: chronological prediction order, never reordered.
    session.currentPresented = presentationQueue.shift();
    session.presentedAt = now;
    session.presentedBeats.push(session.currentPresented);

    if (session.presentedBeats.length > CONFIG.beatStripLength) {
      session.presentedBeats = session.presentedBeats.slice(-CONFIG.beatStripLength);
    }

    renderPresented(session.currentPresented);
    renderBeatsStrip();
  }


  function updatePresentation(payload) {
    const now = performance.now();
    const record = payload.record_name;
    const latestSample = payload.ecg.latest_sample_index;

    // A genuinely new stream (record change, or the sample counter
    // regressing on reconnect) resets all presentation state; the new
    // session's events then enqueue naturally one by one.
    if (session.presentationSeeded) {
      const recordChanged = record !== session.presentationRecord;
      const regressed = latestSample !== null
        && session.presentationLatestSample !== null
        && latestSample < session.presentationLatestSample;

      if (recordChanged || regressed) {
        resetPresentation();
        renderPresented(null);
        renderBeatsStrip();
      }
    }

    session.presentationRecord = record;
    if (latestSample !== null) {
      session.presentationLatestSample = latestSample;
    }

    const events = payload.recent_beats || [];

    if (!session.presentationSeeded) {
      // First payload: seed from the existing bounded history so a
      // mid-stream mount never replays old events. When the stream
      // has not started yet this seeds empty, and the first genuine
      // predictions queue one by one.
      session.presentationSeeded = true;

      for (const event of events) {
        seenPredictionIds.add(record + ":" + event.target_peak_index);
      }

      if (events.length > 0) {
        session.presentedBeats = events.slice(-CONFIG.beatStripLength);
        session.currentPresented = events[events.length - 1];
        session.presentedAt = now;

        // Mid-stream mount: the seeded strip's earlier events become
        // the marker history (consistent with what the strip shows),
        // never the backend's full visible-prediction set.
        for (const event of events.slice(0, -1)) {
          presentedMarkerHistory.push(event);
        }
      }

      renderPresented(session.currentPresented);
      renderBeatsStrip();

      return;
    }

    // Enqueue unseen events exactly once, in chronological order.
    for (const event of events) {
      const id = record + ":" + event.target_peak_index;
      if (seenPredictionIds.has(id)) continue;
      seenPredictionIds.add(id);
      presentationQueue.push(event);
    }

    // Advance at most one event per hold period; when the queue is
    // empty the current event simply stays on screen. Both display
    // modes take this same path - the mode only changes
    // CONFIG.predictionHoldMs, so live simply advances the identical
    // FIFO faster and each transition still gets its own paint.
    if (session.currentPresented === null) {
      if (presentationQueue.length > 0) {
        presentNext(now);
      }
    } else if (presentationQueue.length > 0
               && now - session.presentedAt >= CONFIG.predictionHoldMs) {
      presentNext(now);
    }

    el.classQueue.textContent = presentationQueue.length > 0
      ? "Presenting sequentially \\u00b7 "
        + presentationQueue.length + " queued"
      : "";
  }

  // ---------------------------------------------------------------

  function renderRhythm(payload) {
    el.hr.textContent = fmt(payload.estimated_hr_bpm, 0, " BPM");
    el.rr.textContent = fmt(payload.latest_rr_seconds, 2, " s");
  }

  function renderRuntime(payload) {
    const status = payload.runtime_status;

    if (!status) {
      for (const key of ["rtTemp", "rtCpu", "rtRss", "rtRam", "rtClock",
                         "rtProc", "rtHeadroom", "rtAge",
                         "rtModelLatency", "rtModelThroughput"]) {
        el[key].textContent = UNAVAILABLE;
      }
      el.rtStatusLine.textContent =
        "Waiting for runtime telemetry\\u2026";
      el.rtHistorical.textContent = "";
      return;
    }

    el.rtTemp.textContent = fmt(status.temperature_c, 1, " \\u00B0C");
    el.rtCpu.textContent = fmt(status.process_cpu_percent, 1, "%");
    el.rtRss.textContent = fmt(status.process_rss_mib, 0, " MiB");
    el.rtRam.textContent = fmt(status.available_ram_mib, 0, " MiB");
    el.rtClock.textContent = fmt(status.cpu_frequency_mhz, 0, " MHz");
    el.rtProc.textContent =
      fmt(status.window_max_chunk_processing_ms, 1, " ms");
    el.rtHeadroom.textContent =
      fmt(status.window_min_processing_headroom_ms, 1, " ms");
    el.rtAge.textContent = fmt(payload.runtime_status_age_seconds, 2, " s");
    // Retained model-stage values from the Pi (null before the first
    // inference); fmt renders null as the unavailable dash.
    el.rtModelLatency.textContent =
      fmt(status.model_inference_mean_ms, 2, " ms");
    el.rtModelThroughput.textContent =
      fmt(status.model_throughput_sequences_per_second, 0, " seq/s");

    // Power/thermal text is Python-derived (three-valued semantics);
    // literal throttling uses throttling_active only, never the
    // aggregate condition.
    el.rtStatusLine.textContent =
      "Governor: " + (status.cpu_governor || UNAVAILABLE)
      + "  \\u00B7  Power/thermal: " + status.runtime_condition_text
      + "  \\u00B7  Throttling: " + yesNo(status.throttling_active);
    el.rtHistorical.textContent =
      "Historical power/thermal condition since boot: "
      + yesNo(status.runtime_condition_occurred);
  }

  // ONE DOM update pass per completed fetch: every dynamic panel is
  // refreshed from the same atomic payload. The presentation queue
  // advances BEFORE the chart renders so the single ECG marker and
  // the Classification/Model output panels change in the same pass.
  function render(payload) {
    renderStatus(payload);
    updatePresentation(payload);
    renderEcg(payload);
    renderRhythm(payload);
    renderRuntime(payload);

    if (CONFIG.showDiagnostics) {
      updateDiagnostics();
    }
  }

  function updateDiagnostics() {
    // Optional development diagnostic readout. The production dashboard
    // mounts this component with show_diagnostics=False.
    let meanText = "n/a";
    let maxText = "n/a";
    if (intervals.length > 0) {
      const mean = intervals.reduce(function (a, b) { return a + b; }, 0)
                   / intervals.length;
      meanText = mean.toFixed(0) + " ms";
      maxText = Math.max.apply(null, intervals).toFixed(0) + " ms";
    }
    const sampleText = session.chartState === null
      ? "n/a" : String(session.chartState.latestSampleIndex);
    const presentedText = session.currentPresented === null
      ? "n/a" : String(session.currentPresented.target_peak_index);
    diag.textContent = "updates: " + updateCount
      + " | latest sample: " + sampleText
      + " | requested poll: " + CONFIG.pollMs + " ms"
      + " | measured mean: " + meanText
      + " | max: " + maxText
      + " | queue: " + presentationQueue.length
      + " | presented: " + presentedText
      + " | history: " + presentedMarkerHistory.length;
  }

  async function tick() {
    if (stopped || updating) return;
    updating = true;

    const now = performance.now();
    if (previousTick !== null) {
      intervals.push(now - previousTick);
      if (intervals.length > 50) intervals.shift();
    }
    previousTick = now;

    try {
      const response = await fetch(CONFIG.endpointBase + "/live",
                                   {cache: "no-store"});
      const payload = await response.json();
      if (!stopped) {
        updateCount += 1;
        render(payload);
      }
    } catch (error) {
      if (!stopped) {
        renderWaitingChart("Waiting for ECG stream\\u2026");
      }
    } finally {
      updating = false;
    }
  }

  function start() {
    if (stopped) return;
    tick();
    intervalHandle = setInterval(tick, CONFIG.pollMs);
  }

  function ensurePlotly(onReady) {
    if (window.Plotly) {
      onReady();
      return;
    }

    let script = document.querySelector(
      'script[data-live-ecg-plotly="true"]');

    if (!script) {
      script = document.createElement("script");
      script.src = CONFIG.endpointBase + "/plotly.js";
      script.setAttribute("data-live-ecg-plotly", "true");
      script.onerror = function () {
        diag.textContent =
          "Failed to load Plotly from the local live endpoint.";
      };
      document.head.appendChild(script);
    }

    script.addEventListener("load", onReady);
  }

  // Paint the panels from the session that survived this execution.
  // The bars are rebuilt empty above, so without this a mode change
  // would blank the Classification panel and the class scores until
  // the next prediction advanced. On a genuine first mount there is
  // nothing presented yet and this renders the waiting state, exactly
  // as before.
  renderPresented(session.currentPresented);
  renderBeatsStrip();

  ensurePlotly(start);

  // V2 components are inline: without this cleanup, a page rerun
  // would leak the polling interval (or a pending viewport animation
  // frame) against a detached chart.
  const teardown = () => {
    stopped = true;
    cancelViewportAnimation();
    if (intervalHandle !== null) {
      clearInterval(intervalHandle);
      intervalHandle = null;
    }
  };

  // Returned for Streamlit to call on unmount, AND recorded on the
  // element so a re-execution that never unmounts can stop this
  // instance itself (see the guard at the top).
  parentElement.__liveDashboardTeardown = teardown;

  return teardown;
}
"""

_renderer = None


def live_ecg_config(
    endpoint_base: str,
    poll_ms: int = 100,
    window_seconds: float = 10.0,
    show_diagnostics: bool = True,
    display_mode: str = DEFAULT_DISPLAY_MODE,
) -> dict:
    """
    The mount-time `data` payload for the component (pure, testable).

    Class and connection display metadata come from the presentation
    module, the presentation hold from
    hold_seconds_for_mode (the display mode's only effect), and the
    beat-strip bound from the endpoint's RECENT_BEAT_LIMIT - single
    sources of truth throughout.
    """

    return {
        "endpointBase": endpoint_base,
        "pollMs": int(poll_ms),
        "windowSeconds": float(window_seconds),
        "chartHeight": ECG_FIGURE_HEIGHT,
        "ecgLineColour": ECG_LINE_COLOUR,
        "classOrder": list(CLASS_ORDER),
        "classDisplay": {
            label: {"name": display.name, "colour": display.colour}
            for label, display in CLASS_DISPLAY.items()
        },
        "connectionDisplay": {
            status: {
                "label": connection_label(status),
                "colour": connection_colour(status),
            }
            for status in _CONNECTION_STATUSES
        },
        # The ONLY thing the display mode changes: presentation holds
        # each prediction long enough to read, live advances the same
        # FIFO quickly. The client has one presentation path either
        # way, and the edge pipeline is identical.
        "predictionHoldMs": int(hold_seconds_for_mode(display_mode) * 1000),
        "beatStripLength": RECENT_BEAT_LIMIT,
        "markerHistoryLength": MARKER_HISTORY_LIMIT,
        "viewportAnimationMs": ECG_VIEWPORT_ANIMATION_MS,
        "viewportSnapSeconds": VIEWPORT_SNAP_SECONDS,
        "showDiagnostics": bool(show_diagnostics),
    }


def _live_dashboard_renderer():
    """
    Register the V2 component once per process and cache the renderer.

    Registration is deliberately separate from mounting (per the V2
    docs) and lazy, so importing this module for tests never touches
    Streamlit.
    """

    global _renderer

    if _renderer is None:
        import streamlit as st

        _renderer = st.components.v2.component(
            "live_dashboard",
            html=_HTML,
            css=_CSS,
            js=_JS,
            # Plotly injects its stylesheets into document.head; a
            # shadow root would strand them. The component's own CSS
            # is scoped under #live-dash instead.
            isolate_styles=False,
        )

    return _renderer


def mount_live_dashboard(
    endpoint_base: str,
    poll_ms: int = 100,
    window_seconds: float = 10.0,
    show_diagnostics: bool = True,
    display_mode: str = DEFAULT_DISPLAY_MODE,
) -> None:
    """Mount the live dashboard (call once in the static page body)."""

    _live_dashboard_renderer()(
        key="live_dashboard",
        data=live_ecg_config(
            endpoint_base=endpoint_base,
            poll_ms=poll_ms,
            window_seconds=window_seconds,
            show_diagnostics=show_diagnostics,
            display_mode=display_mode,
        ),
    )
