import threading

import pytest

from ecg_arrhythmia.dashboard.state import (
    CONNECTION_CONNECTED,
    CONNECTION_DISCONNECTED,
    CONNECTION_LISTENING,
    DashboardState,
    DashboardStateConfig,
)

# Small window for tests: 1 second at 10 Hz -> capacity 10 samples.
SMALL_CONFIG = DashboardStateConfig(
    ecg_window_seconds=1.0,
    max_prediction_history=100,
)


def _chunk(start, samples, record="114", rate=10.0):
    return {
        "schema_version": 1,
        "message_type": "sample_chunk",
        "record_name": record,
        "start_index": start,
        "sampling_rate": rate,
        "samples": list(samples),
    }


def _indexed_chunk(start, length, record="114", rate=10.0):
    """Amplitude equals absolute index, so alignment is verifiable."""

    return _chunk(
        start,
        [float(index) for index in range(start, start + length)],
        record=record,
        rate=rate,
    )


def _prediction(target, record="114", label="N"):
    return {
        "schema_version": 1,
        "message_type": "prediction",
        "record_name": record,
        "target_peak_index": target,
        "peak_indices": [target - 10, target, target + 10],
        "logits": [3.5, -1.25, 0.5, -2.0],
        "predicted_class_index": 0,
        "predicted_label": label,
    }


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


# ---------------------------------------------------------------------
#                         Rolling ECG Window
# ---------------------------------------------------------------------


def test_rolling_window_keeps_only_the_latest_samples():
    state = DashboardState(SMALL_CONFIG)

    for start in (0, 4, 8):
        state.apply_message(_indexed_chunk(start, 4))

    snapshot = state.snapshot()

    # 12 samples fed into a 10-sample capacity: the earliest two fell off.
    assert snapshot.sample_indices == tuple(range(2, 12))
    assert snapshot.samples == tuple(float(index) for index in range(2, 12))
    assert len(snapshot.sample_indices) == len(snapshot.samples)


def test_capacity_derives_from_rate_and_window_not_a_constant():
    # 2 seconds at 5 Hz -> capacity 10, nothing resembling 3600.
    state = DashboardState(DashboardStateConfig(ecg_window_seconds=2.0))

    state.apply_message(_indexed_chunk(0, 12, rate=5.0))

    snapshot = state.snapshot()

    assert len(snapshot.samples) == 10
    assert snapshot.sample_indices == tuple(range(2, 12))


def test_contiguous_chunks_produce_no_discontinuities():
    state = DashboardState(SMALL_CONFIG)

    for start in (0, 4, 8):
        state.apply_message(_indexed_chunk(start, 4))

    snapshot = state.snapshot()

    assert snapshot.discontinuities == 0
    assert snapshot.chunks_received == 3
    assert snapshot.samples_received == 12


def test_sample_times_are_derived_from_indices_and_rate():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(2, 3))

    snapshot = state.snapshot()

    assert snapshot.sample_times_seconds() == pytest.approx((0.2, 0.3, 0.4))
    assert snapshot.latest_sample_index == 4


# ---------------------------------------------------------------------
#                       Continuity Protection
# ---------------------------------------------------------------------


def test_a_gap_clears_the_window_and_starts_a_new_region():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4))

    state.apply_message(_indexed_chunk(8, 4))  # expected start was 4

    snapshot = state.snapshot()

    assert snapshot.discontinuities == 1
    # Never a fake continuous trace across the gap: only the new region.
    assert snapshot.sample_indices == tuple(range(8, 12))


def test_an_overlapping_chunk_is_a_discontinuity():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4))

    state.apply_message(_indexed_chunk(2, 4))

    snapshot = state.snapshot()

    assert snapshot.discontinuities == 1
    assert snapshot.sample_indices == tuple(range(2, 6))


def test_a_repeated_chunk_is_a_discontinuity():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4))

    state.apply_message(_indexed_chunk(0, 4))

    snapshot = state.snapshot()

    assert snapshot.discontinuities == 1
    assert snapshot.sample_indices == tuple(range(0, 4))


# ---------------------------------------------------------------------
#                        Record Transitions
# ---------------------------------------------------------------------


def test_a_record_change_clears_the_view_without_joining_waveforms():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(1000, 4, record="114"))
    state.apply_message(_prediction(1002, record="114"))

    state.apply_message(_indexed_chunk(0, 4, record="122"))

    snapshot = state.snapshot()

    assert snapshot.current_record_name == "122"
    # Only the new record's samples are visible.
    assert snapshot.sample_indices == tuple(range(0, 4))
    # Old-record prediction markers and history are gone.
    assert snapshot.recent_predictions == ()
    assert snapshot.visible_predictions == ()
    # A record change is not a stream fault.
    assert snapshot.discontinuities == 0
    # Lifetime counters remain cumulative across records.
    assert snapshot.chunks_received == 2
    assert snapshot.predictions_received == 1


def test_a_sampling_rate_change_never_mixes_time_bases():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4, rate=10.0))

    state.apply_message(_indexed_chunk(4, 4, rate=20.0))

    snapshot = state.snapshot()

    assert snapshot.sampling_rate == 20.0
    # Old-time-base samples were cleared, not combined.
    assert snapshot.sample_indices == tuple(range(4, 8))
    assert snapshot.discontinuities == 1


# ---------------------------------------------------------------------
#                           Predictions
# ---------------------------------------------------------------------


def test_predictions_preserve_wire_fields_and_receive_order():
    state = DashboardState(SMALL_CONFIG)

    for target in (100, 50, 200):
        state.apply_message(_prediction(target))

    recent = state.snapshot().recent_predictions

    # Receive order, never sorted by index.
    assert [prediction.target_peak_index for prediction in recent] == [100, 50, 200]
    first = recent[0]
    assert first.record_name == "114"
    assert first.peak_indices == (90, 100, 110)
    assert first.logits == (3.5, -1.25, 0.5, -2.0)
    assert first.predicted_class_index == 0
    assert first.predicted_label == "N"


def test_prediction_history_is_bounded():
    config = DashboardStateConfig(ecg_window_seconds=1.0, max_prediction_history=3)
    state = DashboardState(config)

    for target in (1, 2, 3, 4, 5):
        state.apply_message(_prediction(target))

    recent = state.snapshot().recent_predictions

    assert [prediction.target_peak_index for prediction in recent] == [3, 4, 5]
    assert state.snapshot().predictions_received == 5


def test_visible_predictions_are_those_inside_the_current_window():
    # 10 s at 100 Hz -> capacity 1000; one chunk spanning 1000..1999.
    config = DashboardStateConfig(ecg_window_seconds=10.0)
    state = DashboardState(config)
    state.apply_message(_chunk(1000, [0.0] * 1000, rate=100.0))

    for target in (1200, 1800, 900, 2100):
        state.apply_message(_prediction(target))

    snapshot = state.snapshot()

    visible = [p.target_peak_index for p in snapshot.visible_predictions]

    assert visible == [1200, 1800]
    assert len(snapshot.recent_predictions) == 4


def test_a_prediction_from_another_record_is_never_visible():
    config = DashboardStateConfig(ecg_window_seconds=10.0)
    state = DashboardState(config)
    state.apply_message(_chunk(1000, [0.0] * 1000, rate=100.0))

    state.apply_message(_prediction(1500, record="999"))

    snapshot = state.snapshot()

    assert snapshot.visible_predictions == ()
    assert len(snapshot.recent_predictions) == 1


# ---------------------------------------------------------------------
#                       Snapshot Semantics
# ---------------------------------------------------------------------


def test_snapshots_are_isolated_from_later_mutation():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4))
    state.apply_message(_prediction(2))

    snapshot_a = state.snapshot()

    state.apply_message(_indexed_chunk(4, 4))
    state.apply_message(_prediction(6))

    assert snapshot_a.sample_indices == tuple(range(0, 4))
    assert snapshot_a.samples == tuple(float(index) for index in range(0, 4))
    assert [p.target_peak_index for p in snapshot_a.recent_predictions] == [2]

    snapshot_b = state.snapshot()

    assert snapshot_b.sample_indices == tuple(range(0, 8))
    assert len(snapshot_b.recent_predictions) == 2


def test_freshness_is_measured_with_the_monotonic_clock():
    clock = FakeClock(100.0)
    state = DashboardState(SMALL_CONFIG, clock=clock)

    assert state.snapshot().last_message_age_seconds is None

    state.apply_message(_indexed_chunk(0, 4))
    clock.value = 102.5

    assert state.snapshot().last_message_age_seconds == pytest.approx(2.5)


# ---------------------------------------------------------------------
#                    Lifecycle, Reset and Errors
# ---------------------------------------------------------------------


def test_connection_lifecycle_states():
    state = DashboardState(SMALL_CONFIG)

    assert state.snapshot().connection_status == CONNECTION_DISCONNECTED

    state.mark_listening()
    assert state.snapshot().connection_status == CONNECTION_LISTENING

    state.mark_client_connected()
    snapshot = state.snapshot()
    assert snapshot.connection_status == CONNECTION_CONNECTED
    assert snapshot.connected

    state.mark_client_disconnected()
    assert state.snapshot().connection_status == CONNECTION_LISTENING

    state.mark_stopped()
    assert state.snapshot().connection_status == CONNECTION_DISCONNECTED


def test_a_new_client_session_clears_the_view_but_not_counters():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4))
    state.apply_message(_prediction(2))

    state.mark_client_connected()

    snapshot = state.snapshot()

    assert snapshot.samples == ()
    assert snapshot.current_record_name is None
    assert snapshot.recent_predictions == ()
    assert snapshot.last_message_age_seconds is None
    assert snapshot.chunks_received == 1
    assert snapshot.predictions_received == 1


def test_reset_stream_optionally_clears_counters():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4))

    state.reset_stream()

    snapshot = state.snapshot()
    assert snapshot.samples == ()
    assert snapshot.sampling_rate is None
    assert snapshot.chunks_received == 1

    state.reset_stream(clear_counters=True)

    assert state.snapshot().chunks_received == 0


def test_record_error_is_exposed_through_the_snapshot():
    state = DashboardState(SMALL_CONFIG)

    state.record_error("connection closed mid-frame")

    assert state.snapshot().last_error == "connection closed mid-frame"


def test_unknown_message_types_raise_rather_than_disappear():
    state = DashboardState(SMALL_CONFIG)

    with pytest.raises(ValueError, match="cannot apply message type 'status'"):
        state.apply_message({"message_type": "status"})


@pytest.mark.parametrize("window", [0, -1.0, float("nan"), float("inf"), True])
def test_invalid_window_configuration_is_rejected(window):
    with pytest.raises(ValueError, match="ecg_window_seconds"):
        DashboardStateConfig(ecg_window_seconds=window)


@pytest.mark.parametrize("history", [0, -5, 2.5, True])
def test_invalid_history_configuration_is_rejected(history):
    with pytest.raises(ValueError, match="max_prediction_history"):
        DashboardStateConfig(max_prediction_history=history)


# ---------------------------------------------------------------------
#                  Runtime Telemetry (Section 6.2.5)
# ---------------------------------------------------------------------


def _runtime_status_message(**overrides):
    message = {
        "schema_version": 2,
        "message_type": "runtime_status",
        "record_name": "114",
        "latest_sample_index": 46217,
        "temperature_c": 48.7,
        "process_cpu_percent": 3.5,
        "process_rss_mib": 253.0,
        "available_ram_mib": 610.0,
        "cpu_frequency_mhz": 2400.0,
        "cpu_governor": "performance",
        "under_voltage_active": False,
        "frequency_capped_active": False,
        "throttling_active": False,
        "soft_temp_limit_active": False,
        "runtime_condition_occurred": False,
        "window_max_chunk_processing_ms": 1.4,
        "window_min_processing_headroom_ms": 98.6,
    }
    message.update(overrides)

    return message


def test_runtime_status_appears_immutably_in_the_snapshot():
    import dataclasses

    state = DashboardState(SMALL_CONFIG)

    assert state.snapshot().runtime_status is None

    state.apply_message(_runtime_status_message())

    snapshot = state.snapshot()
    status = snapshot.runtime_status

    assert status.record_name == "114"
    assert status.latest_sample_index == 46217
    assert status.temperature_c == pytest.approx(48.7)
    assert status.process_cpu_percent == pytest.approx(3.5)
    assert status.process_rss_mib == pytest.approx(253.0)
    assert status.available_ram_mib == pytest.approx(610.0)
    assert status.cpu_frequency_mhz == pytest.approx(2400.0)
    assert status.cpu_governor == "performance"
    assert status.under_voltage_active is False
    assert status.frequency_capped_active is False
    assert status.throttling_active is False
    assert status.soft_temp_limit_active is False
    assert status.runtime_condition_occurred is False
    assert status.window_max_chunk_processing_ms == pytest.approx(1.4)
    assert status.window_min_processing_headroom_ms == pytest.approx(98.6)
    # A v2 message has no model-stage keys at all: absent means "not
    # measured", identical to an explicit null.
    assert status.model_inference_mean_ms is None
    assert status.model_throughput_sequences_per_second is None
    assert status.model_measurement_age_seconds is None
    assert snapshot.runtime_statuses_received == 1

    with pytest.raises(dataclasses.FrozenInstanceError):
        status.temperature_c = 99.9


def test_v3_model_measurements_are_stored_exactly():
    state = DashboardState(SMALL_CONFIG)

    state.apply_message(
        _runtime_status_message(
            schema_version=3,
            model_inference_mean_ms=1.41,
            model_throughput_sequences_per_second=709.2,
            model_measurement_age_seconds=0.0,
        )
    )

    status = state.snapshot().runtime_status

    assert status.model_inference_mean_ms == pytest.approx(1.41)
    assert status.model_throughput_sequences_per_second == pytest.approx(709.2)
    assert status.model_measurement_age_seconds == pytest.approx(0.0)


def test_null_model_measurements_propagate_as_none():
    state = DashboardState(SMALL_CONFIG)

    state.apply_message(
        _runtime_status_message(
            schema_version=3,
            model_inference_mean_ms=None,
            model_throughput_sequences_per_second=None,
            model_measurement_age_seconds=None,
        )
    )

    status = state.snapshot().runtime_status

    assert status.model_inference_mean_ms is None
    assert status.model_throughput_sequences_per_second is None
    assert status.model_measurement_age_seconds is None


def test_only_the_latest_runtime_status_is_retained():
    state = DashboardState(SMALL_CONFIG)

    state.apply_message(_runtime_status_message(temperature_c=48.7))
    state.apply_message(_runtime_status_message(temperature_c=50.1))

    snapshot = state.snapshot()

    assert snapshot.runtime_status.temperature_c == pytest.approx(50.1)
    assert snapshot.runtime_statuses_received == 2


def test_null_hardware_telemetry_fields_survive_into_the_state():
    state = DashboardState(SMALL_CONFIG)

    state.apply_message(
        _runtime_status_message(
            temperature_c=None,
            cpu_frequency_mhz=None,
            cpu_governor=None,
            under_voltage_active=None,
            frequency_capped_active=None,
            throttling_active=None,
            soft_temp_limit_active=None,
            runtime_condition_occurred=None,
        )
    )

    status = state.snapshot().runtime_status

    assert status.temperature_c is None
    assert status.cpu_frequency_mhz is None
    assert status.cpu_governor is None
    assert status.under_voltage_active is None
    assert status.throttling_active is None
    assert status.runtime_condition_occurred is None
    # Every flag unavailable: the derived aggregate is also unknown.
    assert status.runtime_condition_active is None


def test_the_derived_aggregate_condition_reflects_any_active_flag():
    state = DashboardState(SMALL_CONFIG)

    # Under-voltage only: the aggregate warns, but literal throttling
    # stays False - the distinction this design exists for.
    state.apply_message(_runtime_status_message(under_voltage_active=True))

    status = state.snapshot().runtime_status

    assert status.runtime_condition_active is True
    assert status.throttling_active is False

    state.apply_message(_runtime_status_message())

    assert state.snapshot().runtime_status.runtime_condition_active is False


def test_the_derived_aggregate_uses_three_valued_logic():
    state = DashboardState(SMALL_CONFIG)

    # No flag True, one unavailable: an unknown flag could be active,
    # so the aggregate must be unknown rather than a confident "OK".
    state.apply_message(_runtime_status_message(under_voltage_active=None))

    assert state.snapshot().runtime_status.runtime_condition_active is None

    # A definite True dominates regardless of unavailable flags.
    state.apply_message(
        _runtime_status_message(
            under_voltage_active=None,
            frequency_capped_active=None,
            soft_temp_limit_active=None,
            throttling_active=True,
        )
    )

    assert state.snapshot().runtime_status.runtime_condition_active is True


def test_runtime_status_freshness_is_tracked_independently():
    clock = FakeClock(100.0)
    state = DashboardState(SMALL_CONFIG, clock=clock)

    state.apply_message(_runtime_status_message())
    clock.value = 101.0
    state.apply_message(_indexed_chunk(0, 4))
    clock.value = 103.5

    snapshot = state.snapshot()

    # ECG kept flowing after the last status: the two ages diverge.
    assert snapshot.last_message_age_seconds == pytest.approx(2.5)
    assert snapshot.runtime_status_age_seconds == pytest.approx(3.5)


def test_a_new_session_clears_runtime_status_but_not_its_counter():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_runtime_status_message())

    state.mark_client_connected()

    snapshot = state.snapshot()

    assert snapshot.runtime_status is None
    assert snapshot.runtime_status_age_seconds is None
    assert snapshot.runtime_statuses_received == 1


def test_a_record_change_keeps_the_runtime_status():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_indexed_chunk(0, 4, record="114"))
    state.apply_message(_runtime_status_message(record_name="114"))

    state.apply_message(_indexed_chunk(0, 4, record="122"))

    # Same device and process: telemetry continues across records.
    assert state.snapshot().runtime_status is not None


def test_reset_stream_clears_runtime_status_and_optionally_the_counter():
    state = DashboardState(SMALL_CONFIG)
    state.apply_message(_runtime_status_message())

    state.reset_stream()

    assert state.snapshot().runtime_status is None
    assert state.snapshot().runtime_statuses_received == 1

    state.apply_message(_runtime_status_message())
    state.reset_stream(clear_counters=True)

    assert state.snapshot().runtime_statuses_received == 0


# ---------------------------------------------------------------------
#                          Thread Safety
# ---------------------------------------------------------------------


def test_concurrent_writes_and_snapshots_preserve_invariants():
    state = DashboardState(SMALL_CONFIG)
    chunk_length = 4
    total_chunks = 200

    def _writer() -> None:
        for chunk_index in range(total_chunks):
            state.apply_message(
                _indexed_chunk(chunk_index * chunk_length, chunk_length)
            )

    writer = threading.Thread(target=_writer)
    writer.start()

    capacity = 10  # 1 s at 10 Hz

    while writer.is_alive():
        snapshot = state.snapshot()

        # Invariants that must hold in every interleaving.
        assert len(snapshot.sample_indices) == len(snapshot.samples)
        assert len(snapshot.samples) <= capacity

        if snapshot.samples:
            assert snapshot.sampling_rate is not None
            # Amplitude encodes the index, so any corruption or
            # misalignment between the deque and the derived indices
            # would be caught here.
            assert snapshot.samples == tuple(
                float(index) for index in snapshot.sample_indices
            )

    writer.join()

    final = state.snapshot()

    assert final.chunks_received == total_chunks
    assert final.discontinuities == 0
    assert final.sample_indices[-1] == total_chunks * chunk_length - 1
