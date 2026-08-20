import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

KIB_PER_MIB = 1024

PROC_MEMINFO = Path("/proc/meminfo")
PROC_STAT = Path("/proc/stat")
PROC_SELF_STAT = Path("/proc/self/stat")
PROC_SELF_STATUS = Path("/proc/self/status")
THERMAL_ZONE_TEMP = Path("/sys/class/thermal/thermal_zone0/temp")
CPU_GOVERNOR_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
CPU_FREQUENCY_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")

VcgencmdRunner = Callable[[list[str]], str]


def clock_ticks_per_second() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        return 100


# ---------------------------------------------------------------------
#                         vcgencmd Helpers
# ---------------------------------------------------------------------


def _default_vcgencmd_runner(command: list[str]) -> str:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout


def run_vcgencmd(
    argument: str,
    runner: VcgencmdRunner = _default_vcgencmd_runner,
) -> str | None:
    """
    Run one vcgencmd query, returning None wherever it cannot succeed.

    WSL and desktop Linux have no vcgencmd, so absence is a normal,
    silent condition rather than an error.
    """

    try:
        return runner(["vcgencmd", argument]).strip()
    except (OSError, subprocess.SubprocessError):
        logger.debug("vcgencmd %s unavailable", argument)
        return None


def parse_temperature(output: str | None) -> float | None:
    """Parse vcgencmd's "temp=48.2'C" form."""

    # If output is empty, doesn't start with temp= or does not not
    # include ', return None
    if not output or not output.startswith("temp=") or "'" not in output:
        return None

    try:
        # Return only the number. So temp=48.2'C -> 48.2
        return float(output.removeprefix("temp=").split("'")[0])
    except ValueError:
        return None


def parse_throttled(output: str | None) -> str | None:
    """Parse vcgencmd's "throttled=0x0" form, keeping the raw hex flags."""

    if not output or not output.startswith("throttled=0x"):
        return None

    # Returns the hex value, i.e., 0x0 if no throttling has occured.
    # Throttling is when the PI reduces performance due to hardware/
    # power issues. I.e., high temperature, undervoltage, frequency
    # capping, etc.
    return output.removeprefix("throttled=")


# vcgencmd get_throttled bit semantics (Raspberry Pi firmware): bits
# 0-3 are distinct conditions active RIGHT NOW and bits 16-19 their
# sticky "has occurred since boot" counterparts. Each current
# condition is exposed individually so a live indicator can never
# label under-voltage or frequency-capping as literal throttling, and
# a sticky historical bit is never shown as a current condition.
UNDER_VOLTAGE_BIT = 0x1
FREQUENCY_CAPPED_BIT = 0x2
THROTTLED_BIT = 0x4
SOFT_TEMP_LIMIT_BIT = 0x8
OCCURRED_CONDITION_MASK = 0xF0000


def parse_throttled_flags(raw: str | None) -> int | None:
    """The get_throttled bitfield as an integer, None if unavailable."""

    if raw is None:
        return None

    try:
        return int(raw, 16)
    except ValueError:
        return None


def _flag(flags: int | None, mask: int) -> bool | None:
    return None if flags is None else bool(flags & mask)


def under_voltage_active(flags: int | None) -> bool | None:
    """Under-voltage condition right now (bit 0)."""

    return _flag(flags, UNDER_VOLTAGE_BIT)


def frequency_capped_active(flags: int | None) -> bool | None:
    """ARM frequency currently capped (bit 1)."""

    return _flag(flags, FREQUENCY_CAPPED_BIT)


def throttling_active(flags: int | None) -> bool | None:
    """Literal current throttling, bit 2 ONLY - never the other bits."""

    return _flag(flags, THROTTLED_BIT)


def soft_temp_limit_active(flags: int | None) -> bool | None:
    """Soft temperature limit active right now (bit 3)."""

    return _flag(flags, SOFT_TEMP_LIMIT_BIT)


def runtime_condition_occurred(flags: int | None) -> bool | None:
    """Any sticky since-boot occurrence flag set (bits 16-19)."""

    return _flag(flags, OCCURRED_CONDITION_MASK)


# ---------------------------------------------------------------------
#                        /proc and /sys Readers
# ---------------------------------------------------------------------


def parse_meminfo(text: str) -> dict[str, float | None]:
    """
    Extract total and available RAM in MiB from /proc/meminfo text.
    It is interesting to see this doing cat /proc/meminfo
    """

    values: dict[str, float | None] = {
        "total_ram_mib": None,
        "available_ram_mib": None,
    }
    fields = {"MemTotal:": "total_ram_mib", "MemAvailable:": "available_ram_mib"}

    for line in text.splitlines():
        parts = line.split()

        # It needs to be 2 or greater since we need MemTotal: and its corresponding
        # value, and the first one in the list must be what we're looking for
        # in fields.
        if len(parts) >= 2 and parts[0] in fields:
            try:
                values[fields[parts[0]]] = float(parts[1]) / KIB_PER_MIB
            except ValueError:
                pass

    # Returns the total RAM memory and the available ram memory in mib,
    # if it does not exist, it returns None for both.
    return values


def read_meminfo(path: Path = PROC_MEMINFO) -> dict[str, float | None]:
    """Read RAM figures, degrading to None when /proc/meminfo is absent."""

    try:
        return parse_meminfo(path.read_text(encoding="utf-8"))
    except OSError:
        return {"total_ram_mib": None, "available_ram_mib": None}


def read_sysfs_line(path: Path) -> str | None:
    """First line of a sysfs file, or None wherever it cannot be read."""

    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def read_cpu_governor(path: Path = CPU_GOVERNOR_PATH) -> str | None:
    return read_sysfs_line(path)


def read_cpu_frequency_khz(path: Path = CPU_FREQUENCY_PATH) -> int | None:
    value = read_sysfs_line(path)

    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def parse_vmrss_mib(status_text: str) -> float | None:
    """VmRSS in MiB from /proc/self/status content."""

    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()

            try:
                return float(parts[1]) / KIB_PER_MIB
            except (IndexError, ValueError):
                return None

    return None


def read_vmrss_mib(path: Path = PROC_SELF_STATUS) -> float | None:
    try:
        return parse_vmrss_mib(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def parse_proc_stat(stat_text: str) -> tuple[int, int] | None:
    """
    (busy, total) jiffies from the aggregate "cpu " line of /proc/stat.

    Busy is everything except idle and iowait; total includes them.
    """

    for line in stat_text.splitlines():
        if line.startswith("cpu "):
            try:
                fields = [int(value) for value in line.split()[1:]]
            except ValueError:
                return None

            if len(fields) < 5:
                return None

            idle = fields[3] + fields[4]
            total = sum(fields)

            return total - idle, total

    return None


def read_proc_stat(path: Path = PROC_STAT) -> tuple[int, int] | None:
    try:
        return parse_proc_stat(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def parse_process_jiffies(stat_text: str) -> int | None:
    """
    utime + stime from /proc/self/stat.

    The comm field can contain spaces, so fields are counted from the
    closing parenthesis rather than a naive split.
    """

    closing = stat_text.rfind(")")

    if closing < 0:
        return None

    fields = stat_text[closing + 1 :].split()

    # After comm, utime and stime are fields 12 and 13 (0-based).
    try:
        return int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None


def read_process_jiffies(path: Path = PROC_SELF_STAT) -> int | None:
    try:
        return parse_process_jiffies(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def parse_thermal_zone_temp(text: str) -> float | None:
    """Millidegrees from sysfs thermal_zone content, in degrees C."""

    try:
        return int(text.strip()) / 1000.0
    except ValueError:
        return None


def read_temperature_c(
    thermal_path: Path = THERMAL_ZONE_TEMP,
    vcgencmd=run_vcgencmd,
) -> float | None:
    """sysfs thermal zone first (no subprocess), vcgencmd as fallback."""

    text = read_sysfs_line(thermal_path)

    if text is not None:
        temperature = parse_thermal_zone_temp(text)

        if temperature is not None:
            return temperature

    return parse_temperature(vcgencmd("measure_temp"))


# ---------------------------------------------------------------------
#                          CPU Percentages
# ---------------------------------------------------------------------


def system_cpu_percent(
    previous: tuple[int, int] | None,
    current: tuple[int, int] | None,
) -> float | None:
    """Busy fraction of total capacity between two /proc/stat readings."""

    if previous is None or current is None:
        return None

    busy_delta = current[0] - previous[0]
    total_delta = current[1] - previous[1]

    if total_delta <= 0:
        return None

    return busy_delta / total_delta * 100.0


def process_cpu_percent(
    jiffy_delta: int,
    elapsed_seconds: float,
    ticks_per_second: int,
) -> float | None:
    """
    Process CPU as percent of one logical core.

    Values above 100% mean the process kept more than one core busy.
    """

    if elapsed_seconds <= 0 or ticks_per_second <= 0:
        return None

    return jiffy_delta / ticks_per_second / elapsed_seconds * 100.0
