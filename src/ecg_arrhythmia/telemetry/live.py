import logging
import time
from collections.abc import Callable
from pathlib import Path

from ecg_arrhythmia.telemetry.edge_sensors import (
    CPU_FREQUENCY_PATH,
    CPU_GOVERNOR_PATH,
    PROC_MEMINFO,
    PROC_SELF_STAT,
    PROC_SELF_STATUS,
    THERMAL_ZONE_TEMP,
    clock_ticks_per_second,
    frequency_capped_active,
    parse_throttled,
    parse_throttled_flags,
    process_cpu_percent,
    read_cpu_frequency_khz,
    read_cpu_governor,
    read_meminfo,
    read_process_jiffies,
    read_temperature_c,
    read_vmrss_mib,
    run_vcgencmd,
    runtime_condition_occurred,
    soft_temp_limit_active,
    throttling_active,
    under_voltage_active,
)

logger = logging.getLogger(__name__)

KHZ_PER_MHZ = 1000.0


class LiveEdgeTelemetry:
    """
    Sample the live hardware fields of a runtime_status message.

    Paths, the vcgencmd runner and the clock are injectable so tests
    run on fixture files with no Raspberry Pi.
    """

    def __init__(
        self,
        meminfo_path: Path = PROC_MEMINFO,
        process_stat_path: Path = PROC_SELF_STAT,
        process_status_path: Path = PROC_SELF_STATUS,
        thermal_path: Path = THERMAL_ZONE_TEMP,
        governor_path: Path = CPU_GOVERNOR_PATH,
        frequency_path: Path = CPU_FREQUENCY_PATH,
        vcgencmd: Callable[[str], str | None] = run_vcgencmd,
        clock: Callable[[], float] = time.monotonic,
        ticks_per_second: int | None = None,
    ) -> None:
        self.meminfo_path = meminfo_path
        self.process_stat_path = process_stat_path
        self.process_status_path = process_status_path
        self.thermal_path = thermal_path
        self.governor_path = governor_path
        self.frequency_path = frequency_path
        self.vcgencmd = vcgencmd
        self._clock = clock
        self.ticks_per_second = ticks_per_second or clock_ticks_per_second()

        self._previous_jiffies: int | None = None
        self._previous_time: float | None = None
        self._warned_sources: set[str] = set()

    def _note_unavailable(self, source: str) -> None:
        if source in self._warned_sources:
            logger.debug("Telemetry source %s still unavailable", source)

            return

        self._warned_sources.add(source)
        logger.warning(
            "Telemetry source %s unavailable; reporting null",
            source,
        )

    def _process_cpu(self) -> float | None:
        jiffies = read_process_jiffies(self.process_stat_path)
        now = self._clock()
        percent = None

        if (
            jiffies is not None
            and self._previous_jiffies is not None
            and self._previous_time is not None
        ):
            percent = process_cpu_percent(
                jiffies - self._previous_jiffies,
                now - self._previous_time,
                self.ticks_per_second,
            )

        if jiffies is None:
            self._note_unavailable("process CPU (/proc/self/stat)")

        self._previous_jiffies = jiffies
        self._previous_time = now

        return percent

    def sample(self) -> dict:
        """
        One reading of every hardware field, None where unavailable.

        Intended to be called outside any processing timer: this reads
        files and may invoke vcgencmd, and its cost has not been
        benchmarked end to end.
        """

        temperature = read_temperature_c(self.thermal_path, self.vcgencmd)

        if temperature is None:
            self._note_unavailable("temperature (thermal sysfs / vcgencmd)")

        rss = read_vmrss_mib(self.process_status_path)

        if rss is None:
            self._note_unavailable("process RSS (/proc/self/status)")

        available_ram = read_meminfo(self.meminfo_path)["available_ram_mib"]

        if available_ram is None:
            self._note_unavailable("available RAM (/proc/meminfo)")

        frequency_khz = read_cpu_frequency_khz(self.frequency_path)

        if frequency_khz is None:
            self._note_unavailable("CPU frequency (cpufreq sysfs)")

        governor = read_cpu_governor(self.governor_path)

        if governor is None:
            self._note_unavailable("CPU governor (cpufreq sysfs)")

        raw_throttled = parse_throttled(self.vcgencmd("get_throttled"))
        flags = parse_throttled_flags(raw_throttled)

        if flags is None:
            self._note_unavailable("throttling (vcgencmd get_throttled)")
        elif flags:
            # The raw bitfield stays internal; the wire carries the
            # two boolean semantics.
            logger.debug("get_throttled flags: %s", raw_throttled)

        return {
            "temperature_c": temperature,
            "process_cpu_percent": self._process_cpu(),
            "process_rss_mib": rss,
            "available_ram_mib": available_ram,
            "cpu_frequency_mhz": (
                frequency_khz / KHZ_PER_MHZ if frequency_khz is not None else None
            ),
            "cpu_governor": governor,
            "under_voltage_active": under_voltage_active(flags),
            "frequency_capped_active": frequency_capped_active(flags),
            "throttling_active": throttling_active(flags),
            "soft_temp_limit_active": soft_temp_limit_active(flags),
            "runtime_condition_occurred": runtime_condition_occurred(flags),
        }
