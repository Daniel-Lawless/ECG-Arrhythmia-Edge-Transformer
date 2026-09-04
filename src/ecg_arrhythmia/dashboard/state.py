import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CONNECTION_DISCONNECTED = "disconnected"
CONNECTION_LISTENING = "listening"
CONNECTION_CONNECTED = "connected"

DEFAULT_ECG_WINDOW_SECONDS = 10.0
DEFAULT_MAX_PREDICTION_HISTORY = 100


@dataclass(frozen=True)
class DashboardStateConfig:
    """Retention policy for the live view."""

    ecg_window_seconds: float = DEFAULT_ECG_WINDOW_SECONDS
    max_prediction_history: int = DEFAULT_MAX_PREDICTION_HISTORY

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ecg_window_seconds, int | float)
            or isinstance(self.ecg_window_seconds, bool)
            or not math.isfinite(self.ecg_window_seconds)
            or self.ecg_window_seconds <= 0
        ):
            raise ValueError("ecg_window_seconds must be a positive finite number")

        if (
            isinstance(self.max_prediction_history, bool)
            or not isinstance(self.max_prediction_history, int)
            or self.max_prediction_history < 1
        ):
            raise ValueError("max_prediction_history must be a positive integer")


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class ReceivedRuntimeStatus:
    """
    The latest runtime_status message, fields preserved exactly.

    process_cpu_percent keeps the Section 5.4 semantics: percent of
    ONE logical core (400% would mean all four Pi cores busy).
    Hardware fields are None when their source was unavailable on the
    device - never fabricated zeros. The four current power/thermal
    conditions are individual: throttling_active is the literal
    throttling bit only, so an under-voltage-only condition is never
    displayed as throttling; runtime_condition_occurred aggregates the
    sticky since-boot bits and is historical, never current. The
    processing headroom is the nominal chunk period minus the
    interval's largest process_chunk() duration; it is NOT a
    hard-real-time deadline margin (scheduling and network timing are
    excluded).
    """

    record_name: str | None
    latest_sample_index: int | None
    temperature_c: float | None
    process_cpu_percent: float | None
    process_rss_mib: float | None
    available_ram_mib: float | None
    cpu_frequency_mhz: float | None
    cpu_governor: str | None
    under_voltage_active: bool | None
    frequency_capped_active: bool | None
    throttling_active: bool | None
    soft_temp_limit_active: bool | None
    runtime_condition_occurred: bool | None
    window_max_chunk_processing_ms: float
    window_min_processing_headroom_ms: float
    # Live model-stage measurements (schema v3): the mean classifier
    # predict() latency and sequences-per-timed-second throughput of
    # the Pi's most recent inference interval - model-stage capacity,
    # never the ECG prediction/beat rate. None before the first-ever
    # inference or from a pre-v3 sender; quiet intervals carry the
    # retained last valid values, dated by the measurement age.
    model_inference_mean_ms: float | None = None
    model_throughput_sequences_per_second: float | None = None
    model_measurement_age_seconds: float | None = None
    hardware_sample_age_seconds: float | None = None
    hardware_sample_stale: bool | None = None

    @property
    def runtime_condition_active(self) -> bool | None:
        """
        Any current power/thermal condition, derived not transmitted.

        The compact "Power/thermal: OK / Warning" dashboard indicator uses
        three-valued logic: True if any flag is True,
        False only if all four are explicitly False, and None when no
        flag is True but at least one is unavailable - an unknown flag
        could be active, so "OK" would claim more than the evidence
        supports.
        """

        flags = (
            self.under_voltage_active,
            self.frequency_capped_active,
            self.throttling_active,
            self.soft_temp_limit_active,
        )

        if any(flag is True for flag in flags):
            return True

        if all(flag is False for flag in flags):
            return False

        return None


@dataclass(frozen=True)
class ReceivedPrediction:
    """
    One prediction as received over the wire, fields preserved exactly.

    Logits are the raw model output (the classifier stores logits, not
    probabilities, and this layer invents nothing). Immutable, so the
    history deque and any snapshot can share the same instance without
    copies drifting apart.
    """

    record_name: str | None
    target_peak_index: int
    peak_indices: tuple[int, ...]
    logits: tuple[float, ...]
    predicted_class_index: int
    predicted_label: str


@dataclass(frozen=True)
class DashboardSnapshot:
    """
    Immutable view of the live state for one render.

    Every container is a tuple and every element is immutable, so the
    receive thread can keep mutating the live state while the UI holds
    this object. `sample_indices[i]` is the absolute sample position of
    `samples[i]`; visible_predictions are the recent predictions whose
    target index lies inside the current window (and belong to the
    current record).
    """

    connection_status: str
    current_record_name: str | None
    sampling_rate: float | None
    sample_indices: tuple[int, ...]
    samples: tuple[float, ...]
    visible_predictions: tuple[ReceivedPrediction, ...]
    recent_predictions: tuple[ReceivedPrediction, ...]
    chunks_received: int
    samples_received: int
    predictions_received: int
    discontinuities: int
    last_message_age_seconds: float | None
    # Latest edge telemetry: None until the first runtime_status of the
    # session arrives. Its freshness is tracked separately from
    # last_message_age_seconds so "ECG alive but telemetry stalled" is
    # distinguishable from "entire stream stale"; both use PC-side
    # monotonic receipt times, never Pi timestamps.
    runtime_status: ReceivedRuntimeStatus | None
    runtime_status_age_seconds: float | None
    runtime_statuses_received: int
    last_error: str | None = field(default=None)

    @property
    def connected(self) -> bool:
        return self.connection_status == CONNECTION_CONNECTED

    @property
    def latest_sample_index(self) -> int | None:
        return self.sample_indices[-1] if self.sample_indices else None

    def sample_times_seconds(self) -> tuple[float, ...]:
        """Record-relative time per visible sample, derived not stored."""

        if self.sampling_rate is None:
            return ()

        return tuple(index / self.sampling_rate for index in self.sample_indices)


class DashboardState:
    """
    Bounded live state with one writer and any number of snapshotters.

    The receive thread applies decoded protocol messages; the
    dashboard reads immutable snapshots. All mutation and snapshotting
    happens under one short-lived lock.
    """

    def __init__(
        self,
        config: DashboardStateConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config if config is not None else DashboardStateConfig()
        self._clock = clock
        self._lock = threading.Lock()

        self._connection_status = CONNECTION_DISCONNECTED
        self._record_name: str | None = None
        self._sampling_rate: float | None = None

        # Amplitudes of the current contiguous region; capacity derives
        # from sampling_rate x window_seconds once the rate is known.
        self._samples: deque[float] | None = None
        # Absolute index one past the newest retained sample; the first
        # retained index is stop - len(samples), so indices are derived
        # rather than stored per sample.
        self._window_stop_index: int | None = None
        self._expected_next_index: int | None = None

        self._recent_predictions: deque[ReceivedPrediction] = deque(
            maxlen=self._config.max_prediction_history
        )

        # Only the latest runtime status is retained (bounded by
        # design); the dashboard needs current values, not a history.
        self._runtime_status: ReceivedRuntimeStatus | None = None
        self._runtime_status_monotonic: float | None = None

        self._chunks_received = 0
        self._samples_received = 0
        self._predictions_received = 0
        self._discontinuities = 0
        self._runtime_statuses_received = 0

        self._last_message_monotonic: float | None = None
        self._last_error: str | None = None

    @property
    def config(self) -> DashboardStateConfig:
        return self._config

    # -----------------------------------------------------------------
    #                        Message Application
    # -----------------------------------------------------------------

    def apply_message(self, message: dict) -> None:
        """
        Dispatch one decoded protocol message.

        Transport validation already happened in Section 6.1; an
        unknown type here is a programming error and raises rather than
        being swallowed.
        """

        message_type = message["message_type"]

        if message_type == "sample_chunk":
            self.apply_sample_chunk(message)
        elif message_type == "prediction":
            self.apply_prediction(message)
        elif message_type == "runtime_status":
            self.apply_runtime_status(message)
        else:
            raise ValueError(
                f"dashboard state cannot apply message type {message_type!r}"
            )

    def apply_sample_chunk(self, message: dict) -> None:
        record_name = message["record_name"]
        sampling_rate = float(message["sampling_rate"])
        start_index = int(message["start_index"])
        samples = message["samples"]

        with self._lock:
            if self._record_name is not None and record_name != self._record_name:
                # A new record is a new time base, never a continuation:
                # clear the view (including prediction markers, whose
                # indices belong to the old record) without counting a
                # discontinuity.
                logger.info(
                    "Record changed %s -> %s; clearing live view",
                    self._record_name,
                    record_name,
                )
                self._clear_view()
                self._set_sampling_rate(sampling_rate)
            elif self._sampling_rate is None:
                self._set_sampling_rate(sampling_rate)
            elif sampling_rate != self._sampling_rate:
                # Same record, new time base: incompatible with the
                # samples on screen, so treat it as a discontinuity.
                logger.info(
                    "Sampling rate changed %.6g -> %.6g Hz; clearing window",
                    self._sampling_rate,
                    sampling_rate,
                )
                self._discontinuities += 1
                self._clear_window()
                self._set_sampling_rate(sampling_rate)

            self._record_name = record_name

            if (
                self._expected_next_index is not None
                and start_index != self._expected_next_index
            ):
                # Gap, overlap or out-of-order start: never draw a fake
                # continuous trace across it. The window clears and this
                # chunk starts a new contiguous region.
                logger.info(
                    "Stream discontinuity: expected sample %d, received %d",
                    self._expected_next_index,
                    start_index,
                )
                self._discontinuities += 1
                self._clear_window()

            if self._samples is None:
                # Every branch above establishes the window via
                # _set_sampling_rate; this guard keeps that true as a
                # defensive fallback if no earlier path initialised it.
                self._set_sampling_rate(sampling_rate)

            self._samples.extend(float(sample) for sample in samples)
            self._window_stop_index = start_index + len(samples)
            self._expected_next_index = self._window_stop_index

            self._chunks_received += 1
            self._samples_received += len(samples)
            self._last_message_monotonic = self._clock()

        logger.debug(
            "Applied chunk start=%d len=%d record=%s",
            start_index,
            len(samples),
            record_name,
        )

    def apply_prediction(self, message: dict) -> None:
        prediction = ReceivedPrediction(
            record_name=message["record_name"],
            target_peak_index=int(message["target_peak_index"]),
            peak_indices=tuple(int(peak) for peak in message["peak_indices"]),
            logits=tuple(float(logit) for logit in message["logits"]),
            predicted_class_index=int(message["predicted_class_index"]),
            predicted_label=str(message["predicted_label"]),
        )

        with self._lock:
            self._recent_predictions.append(prediction)
            self._predictions_received += 1
            self._last_message_monotonic = self._clock()

        logger.debug(
            "Applied prediction target=%d class=%s",
            prediction.target_peak_index,
            prediction.predicted_label,
        )

    def apply_runtime_status(self, message: dict) -> None:
        """
        Replace the latest runtime status (never appended to a history).

        A record change within one connection does NOT clear this: the
        telemetry describes the same device and process. Only a new
        client session or an explicit reset removes it.
        """

        status = ReceivedRuntimeStatus(
            record_name=message["record_name"],
            latest_sample_index=(
                None
                if message["latest_sample_index"] is None
                else int(message["latest_sample_index"])
            ),
            temperature_c=_optional_float(message["temperature_c"]),
            process_cpu_percent=_optional_float(message["process_cpu_percent"]),
            process_rss_mib=_optional_float(message["process_rss_mib"]),
            available_ram_mib=_optional_float(message["available_ram_mib"]),
            cpu_frequency_mhz=_optional_float(message["cpu_frequency_mhz"]),
            cpu_governor=message["cpu_governor"],
            under_voltage_active=message["under_voltage_active"],
            frequency_capped_active=message["frequency_capped_active"],
            throttling_active=message["throttling_active"],
            soft_temp_limit_active=message["soft_temp_limit_active"],
            runtime_condition_occurred=message["runtime_condition_occurred"],
            window_max_chunk_processing_ms=float(
                message["window_max_chunk_processing_ms"]
            ),
            window_min_processing_headroom_ms=float(
                message["window_min_processing_headroom_ms"]
            ),
            # .get(): a v2 sender omits these entirely, which means
            # exactly "not measured" - the same as an explicit null.
            model_inference_mean_ms=_optional_float(
                message.get("model_inference_mean_ms")
            ),
            model_throughput_sequences_per_second=_optional_float(
                message.get("model_throughput_sequences_per_second")
            ),
            model_measurement_age_seconds=_optional_float(
                message.get("model_measurement_age_seconds")
            ),
            hardware_sample_age_seconds=_optional_float(
                message.get("hardware_sample_age_seconds")
            ),
            hardware_sample_stale=message.get("hardware_sample_stale"),
        )

        with self._lock:
            now = self._clock()
            self._runtime_status = status
            self._runtime_status_monotonic = now
            self._runtime_statuses_received += 1
            self._last_message_monotonic = now

        logger.debug(
            "Applied runtime status: temp=%s governor=%s",
            status.temperature_c,
            status.cpu_governor,
        )

    # -----------------------------------------------------------------
    #                      Connection Lifecycle
    # -----------------------------------------------------------------

    def mark_listening(self) -> None:
        with self._lock:
            self._connection_status = CONNECTION_LISTENING

    def mark_client_connected(self) -> None:
        """
        A new client session began: clear the visible stream state so a
        previous session's waveform is never presented as this one's.
        Lifetime counters remain cumulative.
        """

        with self._lock:
            self._reset_stream_locked(clear_counters=False)
            self._connection_status = CONNECTION_CONNECTED

    def mark_client_disconnected(self) -> None:
        """The client left; the service is back to listening."""

        with self._lock:
            self._connection_status = CONNECTION_LISTENING

    def mark_stopped(self) -> None:
        with self._lock:
            self._connection_status = CONNECTION_DISCONNECTED

    def record_error(self, description: str) -> None:
        """Store a concise description of the most recent stream error."""

        with self._lock:
            self._last_error = description

    # -----------------------------------------------------------------
    #                            Reset
    # -----------------------------------------------------------------

    def reset_stream(self, clear_counters: bool = False) -> None:
        """
        Clear the live view: rolling samples, active record, sampling
        rate, predictions, continuity expectation and freshness.
        Lifetime counters survive unless clear_counters is True.
        """

        with self._lock:
            self._reset_stream_locked(clear_counters=clear_counters)

    def _reset_stream_locked(self, clear_counters: bool) -> None:
        self._clear_view()
        self._record_name = None
        self._sampling_rate = None
        self._samples = None
        self._last_message_monotonic = None
        self._last_error = None
        # A new session must never display the previous session's
        # telemetry as its own.
        self._runtime_status = None
        self._runtime_status_monotonic = None

        if clear_counters:
            self._chunks_received = 0
            self._samples_received = 0
            self._predictions_received = 0
            self._discontinuities = 0
            self._runtime_statuses_received = 0

    def _clear_view(self) -> None:
        self._clear_window()
        self._recent_predictions.clear()

    def _clear_window(self) -> None:
        if self._samples is not None:
            self._samples.clear()

        self._window_stop_index = None
        self._expected_next_index = None

    def _set_sampling_rate(self, sampling_rate: float) -> None:
        if not math.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError(
                f"sampling rate must be positive and finite, got {sampling_rate}"
            )

        capacity = max(
            1,
            math.ceil(sampling_rate * self._config.ecg_window_seconds),
        )
        self._sampling_rate = sampling_rate
        self._samples = deque(maxlen=capacity)
        self._window_stop_index = None
        self._expected_next_index = None

    # -----------------------------------------------------------------
    #                           Snapshot
    # -----------------------------------------------------------------

    def snapshot(self) -> DashboardSnapshot:
        """
        An immutable copy of the live state.

        Copying under the lock is cheap (a few thousand floats and at
        most max_prediction_history predictions); rendering happens
        entirely outside the lock on the returned object.
        """

        with self._lock:
            samples = tuple(self._samples) if self._samples is not None else ()

            if samples and self._window_stop_index is not None:
                first_index = self._window_stop_index - len(samples)
                sample_indices = tuple(range(first_index, self._window_stop_index))
            else:
                sample_indices = ()

            recent = tuple(self._recent_predictions)

            if sample_indices:
                low = sample_indices[0]
                high = sample_indices[-1]
                visible = tuple(
                    prediction
                    for prediction in recent
                    if prediction.record_name == self._record_name
                    and low <= prediction.target_peak_index <= high
                )
            else:
                visible = ()

            now = self._clock()

            if self._last_message_monotonic is None:
                age = None
            else:
                age = now - self._last_message_monotonic

            if self._runtime_status_monotonic is None:
                runtime_age = None
            else:
                runtime_age = now - self._runtime_status_monotonic

            return DashboardSnapshot(
                connection_status=self._connection_status,
                current_record_name=self._record_name,
                sampling_rate=self._sampling_rate,
                sample_indices=sample_indices,
                samples=samples,
                visible_predictions=visible,
                recent_predictions=recent,
                chunks_received=self._chunks_received,
                samples_received=self._samples_received,
                predictions_received=self._predictions_received,
                discontinuities=self._discontinuities,
                last_message_age_seconds=age,
                runtime_status=self._runtime_status,
                runtime_status_age_seconds=runtime_age,
                runtime_statuses_received=self._runtime_statuses_received,
                last_error=self._last_error,
            )
