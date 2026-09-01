import logging

import pytest

from ecg_arrhythmia.telemetry.live import LiveEdgeTelemetry

MEMINFO_TEXT = "MemTotal:        1014464 kB\nMemAvailable:     624640 kB\n"


def _proc_self_stat(utime: int, stime: int) -> str:
    """A /proc/self/stat line whose comm contains a space."""

    fields = ["R"] + [str(value) for value in range(10)]
    fields += [str(utime), str(stime), "0"]

    return f"42 (send record) {' '.join(fields)}"


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _fixture_telemetry(tmp_path, clock, vcgencmd=None):
    (tmp_path / "meminfo").write_text(MEMINFO_TEXT)
    (tmp_path / "self_status").write_text(
        "Name:\tpython3\nVmRSS:\t  258048 kB\nThreads:\t5\n"
    )
    (tmp_path / "self_stat").write_text(_proc_self_stat(500, 500))
    (tmp_path / "thermal").write_text("48700\n")
    (tmp_path / "governor").write_text("performance\n")
    (tmp_path / "frequency").write_text("2400000\n")

    if vcgencmd is None:

        def vcgencmd(argument):
            return {"get_throttled": "throttled=0x0"}.get(argument)

    return LiveEdgeTelemetry(
        meminfo_path=tmp_path / "meminfo",
        process_stat_path=tmp_path / "self_stat",
        process_status_path=tmp_path / "self_status",
        thermal_path=tmp_path / "thermal",
        governor_path=tmp_path / "governor",
        frequency_path=tmp_path / "frequency",
        vcgencmd=vcgencmd,
        clock=clock,
        ticks_per_second=100,
    )


# ---------------------------------------------------------------------
#                        Live Telemetry Sampler
# ---------------------------------------------------------------------


def test_first_sample_reads_hardware_but_reports_no_process_cpu(tmp_path):
    telemetry = _fixture_telemetry(tmp_path, FakeClock(100.0))

    sample = telemetry.sample()

    assert sample["temperature_c"] == pytest.approx(48.7)
    assert sample["process_rss_mib"] == pytest.approx(252.0)
    assert sample["available_ram_mib"] == pytest.approx(610.0)
    assert sample["cpu_frequency_mhz"] == pytest.approx(2400.0)
    assert sample["cpu_governor"] == "performance"
    assert sample["under_voltage_active"] is False
    assert sample["frequency_capped_active"] is False
    assert sample["throttling_active"] is False
    assert sample["soft_temp_limit_active"] is False
    assert sample["runtime_condition_occurred"] is False

    # No previous counters yet: None, never a fabricated zero.
    assert sample["process_cpu_percent"] is None


def test_second_sample_computes_process_cpu_from_jiffy_deltas(tmp_path):
    clock = FakeClock(100.0)
    telemetry = _fixture_telemetry(tmp_path, clock)

    telemetry.sample()

    # 50 more jiffies over 2 seconds at 100 ticks/s -> 25% of one core.
    (tmp_path / "self_stat").write_text(_proc_self_stat(525, 525))
    clock.value = 102.0

    assert telemetry.sample()["process_cpu_percent"] == pytest.approx(25.0)


def test_sticky_throttling_history_is_not_reported_as_active(tmp_path):
    # A brownout ten minutes ago sets sticky bits; the live indicator
    # must not claim any condition is active now.
    def vcgencmd(argument):
        return {"get_throttled": "throttled=0x50000"}.get(argument)

    telemetry = _fixture_telemetry(tmp_path, FakeClock(), vcgencmd=vcgencmd)
    sample = telemetry.sample()

    assert sample["under_voltage_active"] is False
    assert sample["frequency_capped_active"] is False
    assert sample["throttling_active"] is False
    assert sample["soft_temp_limit_active"] is False
    assert sample["runtime_condition_occurred"] is True


def test_under_voltage_only_samples_do_not_claim_throttling(tmp_path):
    def vcgencmd(argument):
        return {"get_throttled": "throttled=0x1"}.get(argument)

    telemetry = _fixture_telemetry(tmp_path, FakeClock(), vcgencmd=vcgencmd)
    sample = telemetry.sample()

    assert sample["under_voltage_active"] is True
    assert sample["throttling_active"] is False


def test_current_throttling_conditions_are_reported_individually(tmp_path):
    def vcgencmd(argument):
        return {"get_throttled": "throttled=0x50005"}.get(argument)

    telemetry = _fixture_telemetry(tmp_path, FakeClock(), vcgencmd=vcgencmd)
    sample = telemetry.sample()

    assert sample["under_voltage_active"] is True
    assert sample["frequency_capped_active"] is False
    assert sample["throttling_active"] is True
    assert sample["soft_temp_limit_active"] is False
    assert sample["runtime_condition_occurred"] is True


def test_unavailable_sources_report_null_and_never_raise(tmp_path):
    telemetry = LiveEdgeTelemetry(
        meminfo_path=tmp_path / "absent_meminfo",
        process_stat_path=tmp_path / "absent_stat",
        process_status_path=tmp_path / "absent_status",
        thermal_path=tmp_path / "absent_thermal",
        governor_path=tmp_path / "absent_governor",
        frequency_path=tmp_path / "absent_frequency",
        vcgencmd=lambda argument: None,
        clock=FakeClock(),
        ticks_per_second=100,
    )

    for _ in range(2):
        sample = telemetry.sample()

        assert sample == {
            "temperature_c": None,
            "process_cpu_percent": None,
            "process_rss_mib": None,
            "available_ram_mib": None,
            "cpu_frequency_mhz": None,
            "cpu_governor": None,
            "under_voltage_active": None,
            "frequency_capped_active": None,
            "throttling_active": None,
            "soft_temp_limit_active": None,
            "runtime_condition_occurred": None,
        }


def test_each_unavailable_source_warns_exactly_once(tmp_path, caplog):
    telemetry = LiveEdgeTelemetry(
        meminfo_path=tmp_path / "absent_meminfo",
        process_stat_path=tmp_path / "absent_stat",
        process_status_path=tmp_path / "absent_status",
        thermal_path=tmp_path / "absent_thermal",
        governor_path=tmp_path / "absent_governor",
        frequency_path=tmp_path / "absent_frequency",
        vcgencmd=lambda argument: None,
        clock=FakeClock(),
        ticks_per_second=100,
    )

    with caplog.at_level(logging.WARNING, logger="ecg_arrhythmia.telemetry.live"):
        telemetry.sample()
        warnings_after_first = [
            record for record in caplog.records if record.levelno == logging.WARNING
        ]

        telemetry.sample()
        warnings_after_second = [
            record for record in caplog.records if record.levelno == logging.WARNING
        ]

    assert len(warnings_after_first) == 7

    # Repeated failures never warn again.
    assert len(warnings_after_second) == len(warnings_after_first)
