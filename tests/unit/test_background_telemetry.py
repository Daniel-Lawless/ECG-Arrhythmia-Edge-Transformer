import threading
from types import SimpleNamespace

import pytest

from ecg_arrhythmia.telemetry.background import HARDWARE_FIELDS, BackgroundEdgeTelemetry


def _hardware() -> dict:
    return {field: None for field in HARDWARE_FIELDS} | {
        "temperature_c": 48.0,
        "process_cpu_percent": 12.0,
        "cpu_governor": "performance",
        "throttling_active": False,
    }


def _assert_null(snapshot: dict) -> None:
    assert snapshot["hardware_sample_stale"] is True
    assert all(snapshot[field] is None for field in HARDWARE_FIELDS)


@pytest.mark.parametrize("field", ["interval_seconds", "max_age_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_configuration_rejects_nonpositive_or_nonfinite_values(field, value):
    with pytest.raises(ValueError, match="positive"):
        BackgroundEdgeTelemetry(None, **{field: value})


def test_cache_reads_never_poll_and_return_independent_snapshots():
    calls = []
    clock = [10.0]
    telemetry = BackgroundEdgeTelemetry(
        SimpleNamespace(sample=lambda: calls.append("poll") or _hardware()),
        clock=lambda: clock[0],
    )

    _assert_null(telemetry.sample())
    assert not calls
    telemetry._collect_once()
    clock[0] = 12.0
    snapshot = telemetry.sample()
    assert snapshot["hardware_sample_age_seconds"] == 2.0
    assert snapshot["hardware_sample_stale"] is False
    snapshot["temperature_c"] = 999
    assert telemetry.sample()["temperature_c"] == 48.0
    assert calls == ["poll"]


def test_collection_age_starts_before_a_slow_poll():
    clock = [0.0]

    def slow_sample():
        clock[0] = 4.0
        return _hardware()

    telemetry = BackgroundEdgeTelemetry(
        SimpleNamespace(sample=slow_sample), clock=lambda: clock[0]
    )
    telemetry._collect_once()

    _assert_null(telemetry.sample())
    assert telemetry.sample()["hardware_sample_age_seconds"] == 4.0


def test_failed_poll_invalidates_cache_and_a_later_poll_recovers():
    calls = 0

    def sample():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("sensor unavailable")
        return _hardware()

    telemetry = BackgroundEdgeTelemetry(SimpleNamespace(sample=sample))
    telemetry._collect_once()
    assert telemetry.sample()["temperature_c"] == 48.0
    telemetry._collect_once()
    _assert_null(telemetry.sample())
    telemetry._collect_once()
    assert telemetry.sample()["temperature_c"] == 48.0
    assert telemetry.failure_count == 1
    assert telemetry.last_error == "sensor unavailable"


def test_blocked_sensor_does_not_block_cache_readers_or_unbounded_shutdown():
    entered = threading.Event()
    release = threading.Event()

    def sample():
        entered.set()
        release.wait(2)
        return _hardware()

    telemetry = BackgroundEdgeTelemetry(SimpleNamespace(sample=sample))
    try:
        telemetry.start()
        assert entered.wait(1)
        _assert_null(telemetry.sample())
        assert telemetry.stop(timeout=0.01) is False
    finally:
        release.set()
        telemetry.stop(timeout=1)

    assert not telemetry._thread.is_alive()
    _assert_null(telemetry.sample())


def test_context_stops_the_collector_after_stream_failure():
    polled = threading.Event()
    telemetry = BackgroundEdgeTelemetry(
        SimpleNamespace(sample=lambda: polled.set() or _hardware()),
        interval_seconds=100,
    )

    with pytest.raises(ValueError, match="stream failed"):
        with telemetry:
            telemetry.start()
            assert polled.wait(1)
            raise ValueError("stream failed")

    assert not telemetry._thread.is_alive()
