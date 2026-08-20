import pytest

from ecg_arrhythmia.telemetry.edge_sensors import (
    frequency_capped_active,
    parse_meminfo,
    parse_proc_stat,
    parse_process_jiffies,
    parse_temperature,
    parse_thermal_zone_temp,
    parse_throttled,
    parse_throttled_flags,
    parse_vmrss_mib,
    process_cpu_percent,
    read_cpu_frequency_khz,
    read_cpu_governor,
    read_sysfs_line,
    runtime_condition_occurred,
    soft_temp_limit_active,
    throttling_active,
    under_voltage_active,
)

MEMINFO_TEXT = "MemTotal:        1014464 kB\nMemAvailable:     624640 kB\n"


def _proc_self_stat(utime: int, stime: int) -> str:
    """A /proc/self/stat line whose comm contains a space."""

    fields = ["R"] + [str(value) for value in range(10)]
    fields += [str(utime), str(stime), "0"]

    return f"42 (send record) {' '.join(fields)}"


# ---------------------------------------------------------------------
#                         Low-Level Parsers
# ---------------------------------------------------------------------


def test_vmrss_parses_to_mib():
    text = "Name:\tpython3\nVmRSS:\t  258048 kB\n"

    assert parse_vmrss_mib(text) == pytest.approx(252.0)
    assert parse_vmrss_mib("Name:\tpython3\n") is None


def test_meminfo_parses_total_and_available_ram():
    values = parse_meminfo(MEMINFO_TEXT)

    assert values["total_ram_mib"] == pytest.approx(990.6875)
    assert values["available_ram_mib"] == pytest.approx(610.0)


def test_thermal_zone_millidegrees_parse_to_celsius():
    assert parse_thermal_zone_temp("48700\n") == pytest.approx(48.7)
    assert parse_thermal_zone_temp("garbage") is None


def test_vcgencmd_temperature_and_throttled_forms_parse():
    assert parse_temperature("temp=48.7'C") == pytest.approx(48.7)
    assert parse_temperature(None) is None
    assert parse_throttled("throttled=0x50000") == "0x50000"
    assert parse_throttled(None) is None


def test_throttling_flags_separate_current_from_sticky_bits():
    assert parse_throttled_flags("0x50005") == 0x50005
    assert parse_throttled_flags(None) is None
    assert parse_throttled_flags("garbage") is None

    # 0x0: nothing active, nothing occurred.
    assert throttling_active(0x0) is False
    assert runtime_condition_occurred(0x0) is False

    # 0x50000: only sticky since-boot bits - no current condition.
    assert under_voltage_active(0x50000) is False
    assert throttling_active(0x50000) is False
    assert runtime_condition_occurred(0x50000) is True

    # Unavailable stays unavailable.
    assert throttling_active(None) is None
    assert runtime_condition_occurred(None) is None


def test_under_voltage_alone_is_never_reported_as_throttling():
    # Bit 0 (under-voltage) set, bit 2 (throttling) clear: the literal
    # throttling flag must stay False.
    flags = 0x1

    assert under_voltage_active(flags) is True
    assert frequency_capped_active(flags) is False
    assert throttling_active(flags) is False
    assert soft_temp_limit_active(flags) is False


def test_each_current_condition_bit_maps_to_its_own_flag():
    assert frequency_capped_active(0x2) is True
    assert throttling_active(0x4) is True
    assert soft_temp_limit_active(0x8) is True

    # 0x50005: under-voltage + throttling now, plus sticky history.
    assert under_voltage_active(0x50005) is True
    assert frequency_capped_active(0x50005) is False
    assert throttling_active(0x50005) is True
    assert runtime_condition_occurred(0x50005) is True


def test_proc_stat_parses_busy_and_total_jiffies():
    text = "cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 25 0 12 200 12 0 0 0 0 0\n"

    assert parse_proc_stat(text) == (150, 1000)


def test_process_jiffies_survive_a_comm_field_with_spaces():
    assert parse_process_jiffies(_proc_self_stat(500, 500)) == 1000


def test_process_cpu_percent_is_percent_of_one_core():
    # 50 jiffies at 100 ticks/s over 2 s -> 25% of one logical core.
    assert process_cpu_percent(50, 2.0, 100) == pytest.approx(25.0)
    assert process_cpu_percent(50, 0.0, 100) is None


def test_sysfs_readers_degrade_to_none(tmp_path):
    present = tmp_path / "governor"
    present.write_text("performance\n")

    frequency = tmp_path / "frequency"
    frequency.write_text("2400000\n")

    assert read_cpu_governor(present) == "performance"
    assert read_cpu_frequency_khz(frequency) == 2400000
    assert read_sysfs_line(tmp_path / "absent") is None
    assert read_cpu_frequency_khz(tmp_path / "absent") is None
