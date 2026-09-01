import json

from ecg_arrhythmia.dashboard.live_ecg_component import (
    _CSS,
    _HTML,
    _JS,
    ECG_VIEWPORT_ANIMATION_MS,
    MARKER_HISTORY_LIMIT,
    VIEWPORT_SNAP_SECONDS,
    ecg_update_decision,
    live_ecg_config,
)
from ecg_arrhythmia.dashboard.live_ecg_server import RECENT_BEAT_LIMIT
from ecg_arrhythmia.dashboard.plots import ECG_FIGURE_HEIGHT, ECG_LINE_COLOUR
from ecg_arrhythmia.dashboard.presentation import (
    CLASS_DISPLAY,
    CLASS_ORDER,
    PREDICTION_PRESENTATION_SECONDS,
    connection_colour,
    connection_label,
)

ENDPOINT = "http://127.0.0.1:8766"


def _payload(record="114", rate=100.0, start=1000, length=100, latest=None):
    return {
        "record_name": record,
        "sampling_rate": rate,
        "ecg": {
            "start_index": start,
            "samples": [0.1] * length,
            "latest_sample_index": (
                latest if latest is not None else start + length - 1
            ),
        },
    }


def _previous(record="114", rate=100.0, latest=1099):
    return {
        "record_name": record,
        "sampling_rate": rate,
        "latest_sample_index": latest,
    }


# ---------------------------------------------------------------------
#                       Mount-Time Configuration
# ---------------------------------------------------------------------


def test_config_payload_carries_the_component_settings():
    config = live_ecg_config(ENDPOINT, poll_ms=100, window_seconds=10.0)

    assert config["endpointBase"] == ENDPOINT
    assert config["pollMs"] == 100
    assert config["windowSeconds"] == 10.0
    assert config["chartHeight"] == ECG_FIGURE_HEIGHT
    assert config["ecgLineColour"] == ECG_LINE_COLOUR
    assert config["classOrder"] == list(CLASS_ORDER)
    # Presentation pacing and strip bound come from single sources.
    assert config["predictionHoldMs"] == int(PREDICTION_PRESENTATION_SECONDS * 1000)
    assert config["beatStripLength"] == RECENT_BEAT_LIMIT
    assert config["markerHistoryLength"] == MARKER_HISTORY_LIMIT
    # One source of truth for the cosmetic viewport glide, finishing
    # before the next genuine ~100 ms chunk arrives.
    assert config["viewportAnimationMs"] == ECG_VIEWPORT_ANIMATION_MS
    assert config["viewportAnimationMs"] < config["pollMs"]
    assert config["viewportSnapSeconds"] == VIEWPORT_SNAP_SECONDS
    # JSON-serialisable: it travels as the V2 mount `data` argument.
    assert json.loads(json.dumps(config)) == config


def test_class_and_connection_display_come_from_the_single_source():
    config = live_ecg_config(ENDPOINT)

    for label, display in CLASS_DISPLAY.items():
        assert config["classDisplay"][label] == {
            "name": display.name,
            "colour": display.colour,
        }

    for status in ("connected", "listening", "disconnected"):
        assert config["connectionDisplay"][status] == {
            "label": connection_label(status),
            "colour": connection_colour(status),
        }


def test_diagnostics_can_be_disabled_after_validation():
    assert live_ecg_config(ENDPOINT, show_diagnostics=True)["showDiagnostics"]
    assert not live_ecg_config(ENDPOINT, show_diagnostics=False)["showDiagnostics"]


# ---------------------------------------------------------------------
#                        V2 Component Source
# ---------------------------------------------------------------------


def test_the_js_is_a_v2_module_reading_mount_data_and_parent_element():
    assert "export default function (component)" in _JS
    assert "const { data, parentElement } = component;" in _JS
    assert "parentElement.querySelector" in _JS
    assert "document.getElementById" not in _JS


def test_one_polling_loop_updates_every_panel_from_one_payload():
    # The central architectural rule: exactly one timer, one fetch of
    # /live per tick, and one DOM pass updating all dynamic panels.
    assert _JS.count("setInterval(") == 1
    assert _JS.count("fetch(") == 1
    assert 'CONFIG.endpointBase + "/live"' in _JS

    for renderer in (
        "renderStatus(payload);",
        "renderEcg(payload);",
        "updatePresentation(payload);",
        "renderRhythm(payload);",
        "renderRuntime(payload);",
    ):
        assert renderer in _JS


def test_the_chart_updates_in_place_with_non_overlapping_polls():
    assert "Plotly.react" in _JS
    assert "Plotly.newPlot" not in _JS
    assert "if (stopped || updating) return;" in _JS
    assert '{cache: "no-store"}' in _JS


def test_the_normal_ecg_path_is_incremental_with_react_as_recovery():
    # Contiguous advances append only the genuinely new samples with
    # front-trimming to the derived window capacity; markers are
    # replaced in place; the moving range uses relayout.
    assert "Plotly.extendTraces(chart, {x: [deltaX], y: [deltaY]}, [0]" in _JS
    assert "Math.round(CONFIG.windowSeconds * rate)" in _JS
    # One shared marker-restyle helper serves the append and hold paths.
    assert _JS.count("Plotly.restyle(") == 1
    assert 'Plotly.relayout(chart, {"xaxis.range"' in _JS
    # Plotly.react remains ONLY for waiting-state and full rebuilds.
    assert _JS.count("Plotly.react(") == 2
    # No interpolation: the delta is a slice of real payload samples.
    assert "samples.slice(deltaOffset)" in _JS


def test_the_update_decision_scenarios():
    # Pure Python mirror of the JS decision (JS authoritative).
    payload = _payload()

    # First render.
    assert ecg_update_decision(None, payload) == "rebuild"
    # Waiting state.
    assert ecg_update_decision(_previous(), _payload(length=0)) == "rebuild"
    # Normal contiguous advancement.
    assert ecg_update_decision(_previous(latest=1050), payload) == "append"
    # Nothing new.
    assert ecg_update_decision(_previous(latest=1099), payload) == "hold"
    # Record change.
    assert ecg_update_decision(_previous(record="122"), payload) == "rebuild"
    # Sampling-rate change.
    assert ecg_update_decision(_previous(rate=360.0), payload) == "rebuild"
    # Sample regression (new session/reconnect).
    assert ecg_update_decision(_previous(latest=5000), payload) == "rebuild"
    # Gap: previous latest no longer overlaps the window.
    assert ecg_update_decision(_previous(latest=500), payload) == "rebuild"


def test_the_ecg_axes_have_no_grid_and_deterministic_second_ticks():
    assert _JS.count("showgrid: false") == 2  # x and y axes
    assert 'tickmode: "linear"' in _JS
    assert "tick0: 0" in _JS
    assert "dtick: 1" in _JS


def test_markers_derive_position_from_the_live_waveform_offset():
    # One shared mapping for current AND historical markers: exact
    # target index against the live window, recomputed every update so
    # every circle stays locked to its beat while scrolling.
    assert "event.target_peak_index - payload.ecg.start_index" in _JS
    assert "payload.ecg.samples[offset]" in _JS
    assert "event.target_peak_index / payload.sampling_rate" in _JS
    # An out-of-window beat produces NO marker: never clamped,
    # edge-pinned or placed on another beat.
    assert "offset < 0 || offset >= payload.ecg.samples.length" in _JS
    assert "Beat no longer visible in ECG window" in _JS


def test_semantic_values_come_from_python_not_javascript():
    # Softmax, RR/HR and the three-valued condition text arrive in the
    # payload; the JS only formats and places them.
    assert "presented.scores" in _JS
    assert "payload.estimated_hr_bpm" in _JS
    assert "payload.latest_rr_seconds" in _JS
    assert "runtime_condition_text" in _JS
    assert "Math.exp" not in _JS
    # Literal throttling uses the literal flag only.
    assert "yesNo(status.throttling_active)" in _JS


def test_the_presentation_queue_is_fifo_seen_once_and_paced_from_config():
    # Explicit queue with stable event identity, FIFO advancement,
    # config-driven hold, seeding on first payload and session reset.
    # The queue and seen-set now live on the persisted session object
    # so a Streamlit rerun cannot restart the presentation engine.
    assert "presentationQueue: []," in _JS
    assert "seenPredictionIds: new Set()," in _JS
    assert 'record + ":" + event.target_peak_index' in _JS
    assert "seenPredictionIds.has(id)" in _JS
    assert "presentationQueue.shift();" in _JS
    assert "now - session.presentedAt >= CONFIG.predictionHoldMs" in _JS
    assert "presentationSeeded" in _JS
    assert "resetPresentation();" in _JS
    # Minimum hold: an empty queue keeps the current event on screen -
    # nothing ever blanks the panel after presentation starts.
    assert "presentationQueue.length > 0" in _JS


def test_ecg_markers_derive_only_from_presented_state():
    # ONE source of truth: the active marker is currentPresented (the
    # same event driving Classification and Model output) and the
    # faded markers are presentedMarkerHistory - events whose turn has
    # ended. visible_predictions never drives any marker, so a queued
    # prediction has NO circle until it is presented.
    assert "function presentedMarkerData(payload)" in _JS
    assert "for (const event of presentedMarkerHistory)" in _JS
    assert "if (session.currentPresented)" in _JS
    assert "payload.visible_predictions" not in _JS
    assert "payload.latest_prediction" not in _JS
    assert "renderPresented(" in _JS
    # Colours come only from the injected class metadata.
    assert "CONFIG.classDisplay[event.predicted_label]" in _JS


def test_the_marker_traces_are_stable_and_styled_by_role():
    # Fixed layout: waveform (0), faded per-point-coloured history (1),
    # single dominant current marker (2) - restyled together, never
    # created/deleted per poll.
    assert (
        '"marker.color": [markers.historical.colours, markers.current.colours],' in _JS
    )
    assert "}, [1, 2]);" in _JS
    # Role distinction is opacity-led: historical markers are faded,
    # the current marker is full opacity with a white outline. Exact
    # pixel sizes are visual tuning and deliberately not pinned here.
    assert "opacity: 0.55}" in _JS
    assert "opacity: 1," in _JS
    assert 'line: {color: "white", width: 1}' in _JS


def test_the_viewport_glides_linearly_and_interruptibly():
    # Ordinary contiguous updates animate the x-range to the genuine
    # latest window via a requestAnimationFrame loop of plain relayout
    # steps - linear by construction, and any in-flight movement is
    # replaced (never queued) when a newer genuine target arrives.
    assert "animateViewportTo(xRangeFor(payload, firstTime));" in _JS
    assert "requestAnimationFrame(step)" in _JS
    assert "cancelAnimationFrame(viewportAnimation)" in _JS
    assert "CONFIG.viewportAnimationMs" in _JS
    # Large/stale jumps (tab stalls, reconnects) snap immediately.
    assert "CONFIG.viewportSnapSeconds" in _JS
    # Plotly's transition engine is deliberately not used: its smooth
    # transitions are documented for SVG scatter only and its frame
    # queue does not fit a 10 Hz replace-while-extending stream.
    assert "Plotly.animate" not in _JS
    assert "easing" not in _JS
    # Recovery renders and unmount always cancel any pending step.
    assert _JS.count("cancelViewportAnimation();") >= 4


def test_no_new_visible_text_was_added_for_the_smoothing():
    # The smoothing is rendering behaviour only: no captions, badges,
    # diagnostics or any other UI text may accompany it.
    lowered = _HTML.lower()

    assert "smooth" not in lowered
    assert "animation" not in lowered
    assert "viewport" not in lowered


def test_the_queue_transition_moves_the_current_marker_into_history():
    # The outgoing event becomes historical in the same presentNext
    # transition that advances the panels - no second timer - and the
    # history is bounded and cleared on stream reset.
    assert "presentedMarkerHistory.push(session.currentPresented);" in _JS
    assert "presentedMarkerHistory.length > CONFIG.markerHistoryLength" in _JS
    assert "presentedMarkerHistory.length = 0;" in _JS


def test_the_marker_legend_is_disabled_rather_than_flickery():
    # The Classification panel already names the class; a legend for a
    # single changing marker would only flicker.
    assert "showlegend: true" not in _JS


def test_the_component_cleans_up_its_polling_interval_on_unmount():
    assert "const teardown = () => {" in _JS
    assert "stopped = true;" in _JS
    assert "cancelViewportAnimation();" in _JS
    assert "clearInterval(intervalHandle)" in _JS
    assert "return teardown;" in _JS


def test_plotly_loads_once_from_the_local_endpoint_never_a_cdn():
    assert 'CONFIG.endpointBase + "/plotly.js"' in _JS
    assert "cdn.plot.ly" not in _JS
    assert "if (window.Plotly)" in _JS


def test_the_html_provides_every_dynamic_panel():
    for element_id in (
        "live-dash",
        "ld-conn-text",
        "ld-record",
        "ld-freshness",
        "ld-gaps",
        "ecg-chart",
        "ld-class-label",
        "ld-class-queue",
        "ld-class-oob",
        "ld-bars",
        "ld-hr",
        "ld-rr",
        "ld-rt-temp",
        "ld-rt-headroom",
        "ld-rt-model-latency",
        "ld-rt-model-throughput",
        "ld-rt-statusline",
        "ld-beats-strip",
    ):
        assert f'id="{element_id}"' in _HTML

    # Score wording stays exact: never confidence/probability.
    assert "Softmax-normalised class scores" in _HTML
    assert "not calibrated probabilities" in _HTML
    assert f"height:{ECG_FIGURE_HEIGHT}px" in _HTML


# ---------------------------------------------------------------------
#              Idempotent Mount (duplicate-row regression)
# ---------------------------------------------------------------------
#
# These are STRUCTURAL assertions on the emitted JavaScript. They pin
# the constructs that make re-execution safe, but they do not execute
# a DOM: that the browser ends up with exactly four rows after
# repeated reruns still needs the manual browser check.


def test_the_model_output_rows_are_cleared_before_being_rebuilt():
    # Regression: a Streamlit rerun re-executes this module against
    # the same persisted DOM, so an unguarded append produced a second
    # N/S/V/F block per rerun (4 -> 8 -> 12 ...).
    build_index = _JS.index("for (const label of CONFIG.classOrder)")
    clear_index = _JS.index("el.bars.replaceChildren();")

    # The existing rows must be cleared before new rows are built.
    assert clear_index < build_index

    # There is exactly one place that clears and constructs the bar rows.
    assert _JS.count("el.bars.replaceChildren();") == 1
    assert _JS.count("el.bars.appendChild(") == 1


def test_the_beat_strip_also_clears_before_appending():
    # The other append-based renderer; it already cleared, and must
    # keep doing so.
    strip_index = _JS.index("function renderBeatsStrip()")
    clear_index = _JS.index('el.beatsStrip.textContent = "";')

    assert strip_index < clear_index
    assert clear_index < _JS.index("el.beatsStrip.appendChild(")


def test_a_previous_instance_is_torn_down_before_a_new_one_starts():
    # Without this, a rerun that re-executes the module without
    # unmounting leaves the old polling loop and its presentation
    # queue running against the same panels.
    guard_index = _JS.index("parentElement.__liveDashboardTeardown()")
    store_index = _JS.index("parentElement.__liveDashboardTeardown = teardown;")

    # The guard runs at mount time, before the teardown is stored.
    assert guard_index < store_index
    # The guard precedes every resource this instance creates.
    assert guard_index < _JS.index("el.bars.replaceChildren();")
    assert guard_index < _JS.index("setInterval(")
    # Still exactly one polling loop per instance.
    assert _JS.count("setInterval(") == 1
    assert "clearInterval(intervalHandle)" in _JS


def test_presentation_state_persists_across_re_execution():
    # Root cause of the "everything updates at once" burst: a mode
    # change re-ran this module, the closure state started empty, and
    # the mid-stream seeding path rebuilt a dozen markers and beats
    # from recent_beats in a single update. The state now lives on the
    # element, so a rerun changes the display policy only.
    assert "parentElement.__liveDashboardSession" in _JS

    setup = _JS[: _JS.index("function fmt(")]

    # Every value the presentation engine carries is created inside
    # the persisted object, not as a fresh closure binding.
    for field in (
        "seenPredictionIds: new Set(),",
        "presentationQueue: [],",
        "presentedMarkerHistory: [],",
        "presentedBeats: [],",
        "currentPresented: null,",
        "presentedAt: null,",
        "presentationSeeded: false,",
        "presentationRecord: null,",
        "presentationLatestSample: null,",
        "chartState: null,",
    ):
        assert field in setup

    # The old per-execution declarations must not come back.
    for declaration in (
        "let currentPresented = null;",
        "let presentedAt = null;",
        "let presentationSeeded = false;",
        "let presentedBeats = [];",
        "let chartState = null;",
    ):
        assert declaration not in _JS


def test_a_persisted_session_is_repainted_on_re_execution():
    # The bars are rebuilt empty, so the surviving prediction must be
    # painted at startup or a mode change would blank the panels until
    # the next advance.
    assert "renderPresented(session.currentPresented);\n  renderBeatsStrip();" in _JS


def test_a_genuine_stream_reset_still_clears_the_persisted_session():
    # Persistence must not defeat the record-change / regression
    # reset: resetPresentation mutates the same object.
    reset_body = _JS[
        _JS.index("function resetPresentation()") : _JS.index(
            "function renderPresented("
        )
    ]

    assert "seenPredictionIds.clear();" in reset_body
    assert "presentationQueue.length = 0;" in reset_body
    assert "presentedMarkerHistory.length = 0;" in reset_body
    assert "session.presentedBeats = [];" in reset_body
    assert "session.currentPresented = null;" in reset_body
    assert "session.presentationSeeded = false;" in reset_body
    # And it is still triggered by the same two conditions.
    assert "const recordChanged = record !== session.presentationRecord;" in _JS
    assert "latestSample < session.presentationLatestSample;" in _JS


def test_seeding_marks_events_seen_without_queueing_them():
    # A remount (including one caused by a mode switch) runs the
    # seeding path. It must record the existing beats as seen and
    # return, never replay them into the queue - otherwise each switch
    # would re-enqueue the whole recent history.
    seed_block = _JS[
        _JS.index("if (!session.presentationSeeded)") : _JS.index(
            "// Enqueue unseen events exactly once"
        )
    ]

    assert "seenPredictionIds.add(record" in seed_block
    assert "presentationQueue.push" not in seed_block
    assert "return;" in seed_block


# ---------------------------------------------------------------------
#                   Display Mode (Presentation / Live)
# ---------------------------------------------------------------------


def test_both_modes_share_one_sequential_presentation_path():
    # The mode changes the dwell and nothing else: no mode branch, no
    # second state machine, no latest-value-wins path in the client.
    assert "displayMode" not in _JS.replace("displayModeBar", "")
    assert "presentLatest" not in _JS
    # No synchronous drain of the queue, in any form.
    assert "while (presentationQueue.length > 0)" not in _JS
    assert "presentationQueue.splice(" not in _JS
    # One transition function, reached from the two arms of the
    # original advance rule.
    assert _JS.count("function presentNext(") == 1
    assert _JS.count("presentNext(now);") == 2
    assert "now - session.presentedAt >= CONFIG.predictionHoldMs" in _JS
    assert "session.currentPresented = presentationQueue.shift();" in _JS
    # The queue is still emptied only by the session reset.
    assert _JS.count("presentationQueue.length = 0;") == 1


def test_the_mode_only_changes_the_prediction_hold():
    from ecg_arrhythmia.dashboard.live_ecg_component import live_ecg_config

    presentation_config = live_ecg_config(endpoint_base="http://127.0.0.1:8766")
    live_config = live_ecg_config(
        endpoint_base="http://127.0.0.1:8766",
        display_mode="Live",
    )

    differing = {
        key
        for key in presentation_config
        if presentation_config[key] != live_config.get(key)
    }

    assert differing == {"predictionHoldMs"}


def test_each_mode_carries_its_own_hold_duration():
    from ecg_arrhythmia.dashboard.live_ecg_component import live_ecg_config
    from ecg_arrhythmia.dashboard.presentation import (
        LIVE_HOLD_SECONDS,
        PREDICTION_PRESENTATION_SECONDS,
    )

    presentation_config = live_ecg_config(endpoint_base="http://127.0.0.1:8766")
    live_config = live_ecg_config(
        endpoint_base="http://127.0.0.1:8766",
        display_mode="Live",
    )

    assert presentation_config["predictionHoldMs"] == int(
        PREDICTION_PRESENTATION_SECONDS * 1000
    )
    assert live_config["predictionHoldMs"] == int(LIVE_HOLD_SECONDS * 1000) == 100
    # Live must remain a real dwell, never zero: each transition needs
    # its own paint rather than collapsing into one.
    assert live_config["predictionHoldMs"] > 0
    assert live_config["predictionHoldMs"] < presentation_config["predictionHoldMs"]


def test_the_css_is_scoped_under_the_component_root():
    for line in _CSS.strip().splitlines():
        stripped = line.strip()

        if stripped.startswith(("#", ".")):
            assert stripped.startswith(("#live-dash", "#ld-", "#ecg-")), line


# ---------------------------------------------------------------------
#           Live Model Metrics (Section 6.3 Final Addition)
# ---------------------------------------------------------------------


def test_exactly_two_model_metrics_exist_with_no_extra_prose():
    # Exactly one label each, in the runtime grid, and nothing else:
    # no explanatory paragraphs, badges, benchmark comparisons or
    # methodology text anywhere in the visible markup.
    assert _HTML.count("Model latency") == 1
    assert _HTML.count("Model throughput") == 1
    assert "Section 5.2" not in _HTML
    assert "benchmark" not in _HTML.lower()
    assert "1.33" not in _HTML  # no hard-coded historical values
    assert "751" not in _HTML


def test_the_model_tooltips_are_title_attributes_only():
    # Clarification lives in hover tooltips, never rendered prose.
    latency_tip = "Mean model-stage latency from the most recent live Pi inference"
    throughput_tip = "not ECG prediction rate."

    assert _HTML.count(latency_tip) == 1
    assert f'title="{latency_tip}' in _HTML
    assert _HTML.count(throughput_tip) == 1
    assert 'title="Active model-stage sequence capacity' in _HTML


def test_the_model_values_render_from_the_payload_with_correct_units():
    # Latency: 2 decimals with ms; throughput: whole seq/s. fmt()
    # renders null as the unavailable dash, so pre-first-inference
    # payloads show the em dash - never 0.00 ms or 0 seq/s.
    assert 'fmt(status.model_inference_mean_ms, 2, " ms")' in _JS
    assert 'fmt(status.model_throughput_sequences_per_second, 0, " seq/s")' in _JS
    # Each payload field drives exactly one element, in renderRuntime.
    assert _JS.count("model_inference_mean_ms") == 1
    assert _JS.count("model_throughput_sequences_per_second") == 1
    # The internal freshness field is deliberately NOT displayed.
    assert "model_measurement_age_seconds" not in _JS


def test_the_model_cells_reset_with_the_rest_of_the_runtime_grid():
    assert '"rtModelLatency", "rtModelThroughput"' in _JS
    assert 'rtModelLatency: $("#ld-rt-model-latency")' in _JS
    assert 'rtModelThroughput: $("#ld-rt-model-throughput")' in _JS


def test_the_runtime_grid_is_five_columns_for_the_model_column():
    assert "repeat(5, 1fr)" in _CSS
    assert "repeat(4, 1fr)" not in _CSS
