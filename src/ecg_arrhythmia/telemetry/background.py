"""Non-blocking cached hardware telemetry for the production stream."""

import logging
import math
import threading
import time

logger = logging.getLogger(__name__)

HARDWARE_FIELDS = (
    "temperature_c",
    "process_cpu_percent",
    "process_rss_mib",
    "available_ram_mib",
    "cpu_frequency_mhz",
    "cpu_governor",
    "under_voltage_active",
    "frequency_capped_active",
    "throttling_active",
    "soft_temp_limit_active",
    "runtime_condition_occurred",
)
DEFAULT_MAX_AGE_SECONDS = 3.0
DEFAULT_STOP_TIMEOUT_SECONDS = 12.0


class BackgroundEdgeTelemetry:
    """Collect hardware readings off the streaming thread and publish snapshots.

    Sensor I/O never holds the cache lock. A cache read is non-blocking and
    returns null hardware fields while the first poll is pending, after an error,
    during lock contention, or when the most recent poll is stale.
    """

    def __init__(
        self,
        source,
        interval_seconds: float = 1.0,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        clock=time.monotonic,
    ) -> None:
        for value in (interval_seconds, max_age_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Telemetry interval and maximum age must be positive")

        self.source = source
        self.interval_seconds = interval_seconds
        self.max_age_seconds = max_age_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot: tuple[float, dict | None] | None = None
        self._thread: threading.Thread | None = None
        self.poll_count = 0
        self.failure_count = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("A telemetry collector cannot be restarted")

        self._thread = threading.Thread(
            target=self._run,
            name="ecg-hardware-telemetry",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._collect_once()
            if self._stop.wait(self.interval_seconds):
                break

    def _collect_once(self) -> None:
        measured_at = self.clock()
        hardware = None

        try:
            values = self.source.sample()
            hardware = {field: values[field] for field in HARDWARE_FIELDS}
        except Exception as error:  # Sensor failures must not stop the stream.
            self.failure_count += 1
            self.last_error = str(error) or type(error).__name__
            if self.failure_count == 1:
                logger.warning("Background hardware collection failed: %s", error)

        self.poll_count += 1
        with self._lock:
            self._snapshot = (measured_at, hardware)

    def sample(self) -> dict:
        """Return the latest complete snapshot without waiting for sensor I/O."""

        snapshot = None
        if self._lock.acquire(blocking=False):
            try:
                snapshot = self._snapshot
            finally:
                self._lock.release()

        age = None if snapshot is None else max(0.0, self.clock() - snapshot[0])
        stale = (
            snapshot is None
            or snapshot[1] is None
            or age > self.max_age_seconds
            or self._stop.is_set()
        )
        values = {field: None for field in HARDWARE_FIELDS}
        if not stale:
            values.update(snapshot[1])

        return {
            **values,
            "hardware_sample_age_seconds": age,
            "hardware_sample_stale": stale,
        }

    def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS) -> bool:
        """Stop and join outside chunk deadlines; report a bounded timeout."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

        return self._thread is None or not self._thread.is_alive()

    def __enter__(self) -> "BackgroundEdgeTelemetry":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self.stop():
            logger.error("Background hardware collector did not stop within timeout")
            if exc_type is None:
                raise RuntimeError("Background hardware collector did not stop")
