import argparse
import json
import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from itertools import cycle, islice
from pathlib import Path
from time import perf_counter_ns, sleep

import numpy as np

from ecg_arrhythmia.data.build_xqrs_centered_dataset import load_split_record_names
from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.evaluation.benchmark_edge_realtime_streaming import (
    chunk_period_ns,
    deadline_statistics,
    run_paced,
    scheduling_statistics,
)
from ecg_arrhythmia.evaluation.benchmark_onnx_inference import (
    environment_metadata,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplaySource
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.streaming.streaming_predictor import StreamingPredictor
from ecg_arrhythmia.telemetry.edge_sensors import (
    PROC_MEMINFO,
    PROC_SELF_STAT,
    PROC_SELF_STATUS,
    PROC_STAT,
    THERMAL_ZONE_TEMP,
    clock_ticks_per_second,
    parse_throttled,
    parse_throttled_flags,
    process_cpu_percent,
    read_cpu_frequency_khz,
    read_cpu_governor,
    read_meminfo,
    read_proc_stat,
    read_process_jiffies,
    read_temperature_c,
    read_vmrss_mib,
    run_vcgencmd,
    system_cpu_percent,
)

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_SUMMARY = Path("data/splits_sequences_matched/split_summary_metrics.json")
DEFAULT_FP32_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_INT8_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer_int8.onnx")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/deployment_evaluation/edge_sustained_resources"
)

DEFAULT_DURATION_SECONDS = 3600.0
DEFAULT_MONITOR_INTERVAL_SECONDS = 5.0
EXPECTED_GOVERNOR = "performance"

NANOSECONDS_PER_SECOND = 1_000_000_000
SECONDS_PER_HOUR = 3600.0

# Trailing window used for the reported final temperature mean.
THERMAL_FINAL_WINDOW_SECONDS = 600.0

MAX_RECORDED_MISSES = 50


# ---------------------------------------------------------------------
#                        Telemetry Sampling
# ---------------------------------------------------------------------


class ProgressTracker:
    """
    Lightweight streaming progress shared with the sampler thread.
    tracks how far we are through a record so telementry data can be
    saved with more context
    """

    def __init__(self) -> None:
        self.record_name: str | None = None
        self.total_chunks = 0


def tracked_chunks(
    chunks: Iterable,
    progress: ProgressTracker,
) -> Iterator:
    """Count chunks as they are drawn, outside the timed boundary."""

    for chunk in chunks:
        progress.total_chunks += 1
        yield chunk


class TelemetryReader:
    """
    One point-in-time telemetry sample, with CPU deltas across calls.

    Every reader is injectable, so tests supply fake content and never
    touch real hardware. The first sample has no previous counters, so
    its CPU percentages are None.
    """

    def __init__(
        self,
        meminfo_path: Path = PROC_MEMINFO,
        proc_stat_path: Path = PROC_STAT,
        process_stat_path: Path = PROC_SELF_STAT,
        process_status_path: Path = PROC_SELF_STATUS,
        thermal_path: Path = THERMAL_ZONE_TEMP,
        vcgencmd=run_vcgencmd,
        frequency_reader=read_cpu_frequency_khz,
        governor_reader=read_cpu_governor,
        ticks_per_second: int | None = None,
    ) -> None:
        self.meminfo_path = meminfo_path
        self.proc_stat_path = proc_stat_path
        self.process_stat_path = process_stat_path
        self.process_status_path = process_status_path
        self.thermal_path = thermal_path
        self.vcgencmd = vcgencmd
        self.frequency_reader = frequency_reader
        self.governor_reader = governor_reader
        self.ticks_per_second = ticks_per_second or clock_ticks_per_second()

        self._previous_system: tuple[int, int] | None = None
        self._previous_process_jiffies: int | None = None
        self._previous_elapsed: float | None = None

    def sample(self, elapsed_seconds: float, progress: ProgressTracker) -> dict:
        # Extracts total and available ram
        memory = read_meminfo(self.meminfo_path)

        # CPU usage of the whole raspberry PI. The system counters only ever increase
        # which is why we can see if CPU utilization has increased or decreased over
        # a given time. For example, one telementry sample might give
        # total CPU time = 100,000 ticks, idle CPU time = 80,000 ticks, then on the
        # next sample we get total CPU time = 102,000 ticks idle CPU time = 81,000 ticks
        # so during that interval total increase = 2000, idle increase = 1000,
        # busy increase = 1000, so CPU usage = busy increase / total increase
        # = 1000 / 2000 = 50%. This is CPU usage over the whole PI.
        system_counters = read_proc_stat(self.proc_stat_path)

        # CPU time consumed by THIS Python ECG program since we're reading
        # proc/self/stat. A jiffy for Linux is essentially a CPU-time tick.
        # we then convert those time ticks into seconds by using the systems
        # clock tick per second value.
        process_jiffies = read_process_jiffies(self.process_stat_path)

        # Calculate the CPU usage by comparing the system counters
        # at this point to the system counter of the last sample
        system_percent = system_cpu_percent(
            self._previous_system,
            system_counters,
        )
        process_percent = None

        if (
            process_jiffies is not None
            and self._previous_process_jiffies is not None
            and self._previous_elapsed is not None
        ):
            # calculates CPU usage for our Python ECG process
            process_percent = process_cpu_percent(
                # How many extra CPU jiffies have we accumulated since
                # the last sample
                process_jiffies - self._previous_process_jiffies,
                # How much time has passed since the last sample
                elapsed_seconds - self._previous_elapsed,
                # How may time ticks per second is Linux using
                self.ticks_per_second,
            )

        # This sample now becomes the previous sample.
        self._previous_system = system_counters
        self._previous_process_jiffies = process_jiffies
        self._previous_elapsed = elapsed_seconds

        # Extract the throttle flag for this sample
        throttled = self.vcgencmd("get_throttled")

        # Return a summary of this telementry sample
        return {
            "elapsed_seconds": elapsed_seconds,
            "temperature_c": read_temperature_c(
                self.thermal_path,
                self.vcgencmd,
            ),
            "available_ram_mib": memory["available_ram_mib"],
            "total_ram_mib": memory["total_ram_mib"],
            "rss_mib": read_vmrss_mib(self.process_status_path),
            "system_cpu_percent": system_percent,
            "process_cpu_percent": process_percent,
            "cpu_frequency_khz": self.frequency_reader(),
            "cpu_governor": self.governor_reader(),
            "throttled": parse_throttled(throttled),
            "record_name": progress.record_name,
            "total_chunks_processed": progress.total_chunks,
        }


class TelemetrySampler(threading.Thread):
    """
    Daemon thread sampling telemetry at a fixed interval.

    The stop event doubles as the interval timer, so the thread wakes
    immediately when asked to stop. Sampling failures are counted and
    surfaced rather than silently swallowed or allowed to kill the run.
    """

    def __init__(
        self,
        reader: TelemetryReader,
        progress: ProgressTracker,
        interval_seconds: float,
        clock=perf_counter_ns,
        stop_event: threading.Event | None = None,
    ) -> None:
        # This is satisfying the Thread class we're inheriting from.
        # daemon=True makes it so if the main program ends, and this
        # thread is still running, then Python will exit.
        super().__init__(daemon=True)
        self.reader = reader
        self.progress = progress
        self.interval_seconds = interval_seconds
        self.clock = clock
        # threading.Event(): an event is a thread-safe boolean flag
        # when we first create this event = off
        self.stop_event = self.stop_event = (
            stop_event if stop_event is not None else threading.Event()
        )
        self.samples: list[dict] = []
        self.failure_count = 0
        self.failures: list[str] = []
        self._start_ns: int | None = None

    # When we inherit from threading.Thread, we are provided with functions
    # like is_alive(), start(), join(), etc. We also need to override run()
    # since when this new thread is created, it needs to know what code to run.
    # We don't actually call this function, when we run sampler.start(),
    # this thread will be created and will run this code.
    def run(self) -> None:
        # This is when the telementry thread starts
        self._start_ns = self.clock()

        while True:
            # Reads the telementry data
            self.sample_once()

            # wait() is a function from threading_Event(). It means
            # pause this thread until the Event becomes set, if
            # interval_seconds is 5, then we wait for 5 seconds for it to
            # be set, else we move back through the loop and
            # sample again.
            if self.stop_event.wait(self.interval_seconds):
                return

    def sample_once(self) -> None:
        # Calculates how many seconds have passed since the telemetry thread started
        total_elapsed = (self.clock() - self._start_ns) / NANOSECONDS_PER_SECOND

        try:
            # Reads a telementry sample and appends it to self.samples
            self.samples.append(self.reader.sample(total_elapsed, self.progress))
        except Exception as error:  # noqa: BLE001 - surfaced, not hidden
            # Increments failure count
            self.failure_count += 1

            # If we have less than 5 failures, then append the whole failure,
            # else we just append the failure count and move on
            if len(self.failures) < 5:
                self.failures.append(f"t={total_elapsed:.1f}s: {error!r}")

    def stop(self, timeout_seconds: float = 10.0) -> None:
        # sets the threading.Event to True / set. Once set is called,
        # wait() wakes up and returns true, at which case we return, and
        # the telementry thread starts to shutdown
        self.stop_event.set()

        # Checks if the telemetry thread is still running right now
        if self.is_alive():
            # If it is still alive, then join() means wait for that
            # telemetry thread to finish, for up to timeout_seconds seconds.
            self.join(timeout=timeout_seconds)


# ---------------------------------------------------------------------
#                       Telemetry Analysis
# ---------------------------------------------------------------------


def _valid_pairs(
    elapsed: list[float],
    values: list[float | None],
) -> tuple[np.ndarray, np.ndarray]:
    # Groups the time values and corresponding sample values into a tuple
    # and only keeps the tuple if the value is not None and is finite.
    pairs = [
        (time, value)
        for time, value in zip(elapsed, values, strict=True)
        if value is not None and np.isfinite(value)
    ]

    if not pairs:
        return np.array([]), np.array([])

    # Cool trick to unzip the pairs back to the original elasped
    # and values arrays but now the values are cleaned.
    times, cleaned = zip(*pairs, strict=True)

    return (np.asarray(times, dtype=np.float64), np.asarray(cleaned, dtype=np.float64))


def series_summary(values: list[float | None]) -> dict | None:
    """Start/end/min/mean/max/delta over the valid points of a series."""

    # Only keeps values that are not None and are finite.
    valid = [value for value in values if value is not None and np.isfinite(value)]

    if not valid:
        return None

    return {
        "start": valid[0],
        "end": valid[-1],
        "minimum": float(min(valid)),
        "mean": float(np.mean(valid)),
        "maximum": float(max(valid)),
        "delta": valid[-1] - valid[0],
    }


def slope_per_hour(
    elapsed: list[float],
    values: list[float | None],
) -> float | None:
    """
    Estimate the linear rate of change in the telemetry values per hour.

    Fits a straight line through the valid samples and returns its slope
    converted from units per second to units per hour.
    """

    # Remove invalid values while keeping each value matched to its timestamp.
    times, cleaned = _valid_pairs(elapsed, values)

    # At least two valid samples at different times are needed to fit a line.
    if times.size < 2 or times[-1] == times[0]:
        return None

    # Fit a straight line through the telemetry samples.
    # np.polyfit(..., 1) returns [slope, intercept], so [0] is the slope.
    slope_per_second = np.polyfit(times, cleaned, 1)[0]

    # elapsed is measured in seconds, so convert the slope to units per hour.
    return float(slope_per_second * SECONDS_PER_HOUR)


def final_window_mean(
    elapsed: list[float],
    values: list[float | None],
    window_seconds: float = THERMAL_FINAL_WINDOW_SECONDS,
) -> float | None:
    """
    Calculates the average of the valid telemetry values
    from the final window_seconds of the run
    """

    # returns only valid values along with their corresponding elasped
    # times
    times, cleaned = _valid_pairs(elapsed, values)

    if times.size == 0:
        return None

    # we want the final window seconds of the run. So we take the
    # last telementry run, and subtract window_seconds from it.
    cutoff = times[-1] - window_seconds

    # keeps only the cleaned values where times >= cutoff, then
    # we take the mean of those remaining values.
    return float(cleaned[times >= cutoff].mean())


def time_to_maximum(
    elapsed: list[float],
    values: list[float | None],
) -> float | None:
    # Returns only valid values with its corresponding elasped times
    times, cleaned = _valid_pairs(elapsed, values)

    if times.size == 0:
        return None

    # Calculates the index of the largest value
    index = int(np.argmax(cleaned))

    # Returns the elasped time that this largest value occured at
    return float(times[index])


def throttling_summary(
    elapsed: list[float],
    throttled_values: list[str | None],
) -> dict:
    """Whether the hardware ever reported a non-clean throttled state."""

    observed = [
        (time, value)
        for time, value in zip(elapsed, throttled_values, strict=True)
        if value is not None
    ]
    # if value is not 0x0, it is a throttling state, so it is not clean
    # and throttling has occured
    non_clean = [(time, value) for time, value in observed if value != "0x0"]
    known_values = [value for _, value in observed]

    return {
        "samples_with_readings": len(observed),
        "any_throttling_observed": bool(non_clean),
        # Extracts the time from the first tuple
        "first_throttling_elapsed_seconds": (non_clean[0][0] if non_clean else None),
        "unique_values": sorted(set(known_values)),
        "final_value": known_values[-1] if known_values else None,
    }


def correlate_misses(misses: list[dict], samples: list[dict]) -> list[dict]:
    """
    For every deadline miss, find the telemetry sample taken closest to
    when that miss happened, and attach its Pi resource information to the miss.
    """

    if not samples:
        return misses

    # Return the start time of each telementry sample
    sample_times = np.asarray(
        [sample["elapsed_seconds"] for sample in samples],
        dtype=np.float64,
    )
    correlated = []

    for miss in misses:
        # Calculate difference between each sample time and the time at which this miss
        # occured.
        differences = np.abs(sample_times - miss["elapsed_seconds"])

        # Find the index where the smallest difference occurs. This telemetry sample
        # was the closest when the miss occured.
        index = int(np.argmin(differences))

        # Return the telementry sample associated with this miss
        nearest = samples[index]

        # Append this to our correlated list. This will map one-to-one with our
        # misses, giving us the closest telementry sample for each miss
        correlated.append(
            {
                **miss,
                "nearest_telemetry": {
                    "elapsed_seconds": nearest["elapsed_seconds"],
                    "temperature_c": nearest["temperature_c"],
                    "cpu_frequency_khz": nearest["cpu_frequency_khz"],
                    "rss_mib": nearest["rss_mib"],
                    "available_ram_mib": nearest["available_ram_mib"],
                },
            }
        )

    return correlated


# ---------------------------------------------------------------------
#                           Model Warm-Up
# ---------------------------------------------------------------------


def warm_up_predictor(
    predictor: StreamingPredictor,
    source: ReplaySource,
) -> dict:
    """
    Exercise the real streaming path until ONNX inference has happened.

    This happens before telemetry starts, so one-time ONNX Runtime setup
    does not distort the sustained-run latency/resource measurements.
    The predictor is reset afterwards so no detector, sequence or record
    state from the warm-up can leak into the measured run.
    """

    # Start a temporary record solely for the warm-up.
    predictor.start_record(record_name=f"{source.record_name}_warmup")

    chunks_processed = 0
    predictions = 0

    try:
        # Feed real ECG chunks through the full production pipeline until
        # at least one complete sequence reaches the ONNX classifier.
        for chunk in source.iter_chunks():
            chunks_processed += 1
            predictions += len(predictor.process_chunk(chunk))

            if predictions > 0:
                break

        # If no prediction was emitted while streaming the record, give the
        # end-of-record path a chance to release a final complete sequence.
        if predictions == 0:
            predictions += len(predictor.flush())
    finally:
        # Discard every bit of streaming state created by the warm-up so the
        # measured sustained run starts from a completely clean record state.
        predictor.reset()

    if predictions == 0:
        raise RuntimeError(
            f"Warm-up record {source.record_name!r} produced no predictions."
        )

    return {
        "record_name": source.record_name,
        "chunks_processed": chunks_processed,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------
#                     Sustained Streaming Run
# ---------------------------------------------------------------------


def sustained_streaming_run(
    predictor,
    record_names: list[str],
    source_factory: Callable,
    duration_ns: int,
    chunk_size: int,
    progress: ProgressTracker,
    clock=perf_counter_ns,
    sleeper=sleep,
) -> dict:
    """
    Cycle through validation records in real time until the requested
    signal duration is reached.

    Each record starts with clean streaming state, and the final record
    is truncated if only part of it fits in the remaining duration.
    Per-record timing data is summarised before moving to the next record.
    """

    overall_start_ns = clock()
    consumed_signal_ns = 0
    per_record: list[dict] = []
    misses: list[dict] = []

    totals = {
        "chunks": 0,
        "predictions": 0,
        "flush_predictions": 0,
        "deadline_misses": 0,
        "integrity_failures": 0,
        "max_processing_latency_ms": 0.0,
        "max_scheduling_lateness_ms": 0.0,
        "final_scheduling_lateness_ms": None,
        "minimum_deadline_margin_ms": None,
    }

    # cycle(record_names) creates a cycle object that we can iterate through.
    # It allows us to cycle through the list of record names
    for record_name in cycle(record_names):
        # Calculates how much time we have left of this sustained run in ns
        remaining_ns = duration_ns - consumed_signal_ns

        # Returns the replay souce for this record
        source = source_factory(record_name)

        # Extracts how many nanoseconds a chunk represents
        period_ns = chunk_period_ns(chunk_size, source.sampling_rate)

        # Performs floor divison to get the number of chunks represented by the
        # remaining seconds
        max_chunks = int(remaining_ns // period_ns)

        # Do not start another record unless at least two chunks fit in the
        # remaining duration. A one-chunk paced record ends immediately after
        # chunk 0 because the first chunk arrives at t=0, so it does not
        # meaningfully exercise the pacing behaviour. It is also  too short
        # to produce output through the detector/sequence warm-up.
        if max_chunks < 2:
            break

        # Can the entire record chunks be fit in the remaining chunks?
        truncated = max_chunks < source.num_chunks

        # Sets the record field for our progress checker
        progress.record_name = record_name

        # Gets the stremaing predictor ready to process a new record.
        predictor.start_record(record_name=record_name)

        # Run min(source.num_chunks, max_chunks) through the paced run,
        # extracting metrics like start times, scheduled times, etc.
        run = run_paced(
            predictor,
            tracked_chunks(
                islice(source.iter_chunks(), max_chunks),
                progress,
            ),
            period_ns=period_ns,
            clock=clock,
            sleeper=sleeper,
        )

        # Returns deadline and scheduling stats
        deadline = deadline_statistics(
            run["scheduled_ns"],
            run["completion_ns"],
            period_ns,
        )
        lateness = scheduling_statistics(
            run["scheduled_ns"],
            run["actual_start_ns"],
        )
        # extract the accumulator used for this record
        accumulator = run["accumulator"]

        # Misses are resolved to overall elapsed time now, before the
        # per-chunk arrays are discarded.
        lateness_ns = np.asarray(run["completion_ns"], dtype=np.int64) - (
            np.asarray(run["scheduled_ns"], dtype=np.int64) + period_ns
        )

        # returns the index of each missed deadline
        for index in np.flatnonzero(lateness_ns > 0):
            if len(misses) >= MAX_RECORDED_MISSES:
                break

            # Append information about the misses
            misses.append(
                {
                    "record_name": record_name,
                    "chunk_index": int(index),
                    "elapsed_seconds": (
                        run["completion_ns"][int(index)] - overall_start_ns
                    )
                    / NANOSECONDS_PER_SECOND,
                    "deadline_lateness_ms": float(lateness_ns[int(index)]) / 1_000_000,
                }
            )
        # append information about this record
        per_record.append(
            {
                "record_name": record_name,
                "truncated": truncated,
                "chunks_processed": len(run["processing_ns"]),
                "samples_processed": (predictor.engine.state.total_samples_accepted),
                "predictions": accumulator.num_events,
                "flush_predictions": accumulator.flush_event_count,
                "record_wall_seconds": run["paced_wall_ns"] / NANOSECONDS_PER_SECOND,
                "deadline_misses": deadline["deadline_misses"],
                "max_processing_latency_ms": max(
                    duration / 1_000_000 for duration in run["processing_ns"]
                ),
                "integrity_passed": accumulator.integrity_passed,
                "class_counts": dict(accumulator.class_counts),
            }
        )

        # Update the running totals
        totals["chunks"] += len(run["processing_ns"])
        totals["predictions"] += accumulator.num_events
        totals["flush_predictions"] += accumulator.flush_event_count
        totals["deadline_misses"] += deadline["deadline_misses"]
        totals["integrity_failures"] += accumulator.integrity_failure_count
        totals["max_processing_latency_ms"] = max(
            totals["max_processing_latency_ms"],
            per_record[-1]["max_processing_latency_ms"],
        )
        totals["max_scheduling_lateness_ms"] = max(
            totals["max_scheduling_lateness_ms"],
            lateness["maximum"],
        )
        totals["final_scheduling_lateness_ms"] = lateness["final"]
        margin = -deadline["maximum_deadline_lateness_ms"]

        if (
            totals["minimum_deadline_margin_ms"] is None
            or margin < totals["minimum_deadline_margin_ms"]
        ):
            totals["minimum_deadline_margin_ms"] = margin

        # Get the number of samples that have been run through the engine for
        # this record.
        record_samples = predictor.engine.state.total_samples_accepted
        # Update the consumed signal, this is how much signal has currently
        # been processed.
        consumed_signal_ns += round(
            record_samples / source.sampling_rate * NANOSECONDS_PER_SECOND
        )

        # Explicitly release the previous record's temporary objects before
        # constructing the next record's temporary objects, avoiding even
        # transient overlap and keeping the sustained-memory profile as clean
        # as possible.
        del run, source

        logger.info(
            "Completed %s record %s (%d chunks, %d misses)",
            "truncated" if truncated else "full",
            record_name,
            per_record[-1]["chunks_processed"],
            per_record[-1]["deadline_misses"],
        )

    # time at the end - time at the beginning to get the total in nanoseconds,
    # then convert to seconds.
    actual_duration = (clock() - overall_start_ns) / NANOSECONDS_PER_SECOND

    return {
        "per_record": per_record,
        "totals": totals,
        "misses": misses,
        "actual_duration_seconds": actual_duration,
        "paced_signal_seconds": consumed_signal_ns / NANOSECONDS_PER_SECOND,
    }


# ---------------------------------------------------------------------
#                       Summaries And Status
# ---------------------------------------------------------------------


def telemetry_column(samples: list[dict], key: str) -> list:
    return [sample[key] for sample in samples]


def rss_trend(
    elapsed: list[float],
    values: list[float | None],
) -> dict:
    """
    Estimate the overall RSS memory trend across the sustained run.

    Fits a straight line through the valid RSS samples and reports the
    rate of change, total fitted change, and scatter around the trend.
    """

    # Clean the times and values such that only valid values have
    # a corresponding time
    times, cleaned = _valid_pairs(elapsed, values)

    # At least three points are needed to estimate both a linear trend
    # and meaningful scatter around that trend.
    if times.size < 3 or times[-1] == times[0]:
        return {
            "status": "insufficient_data",
            "slope_mib_per_hour": None,
            "fitted_change_mib": None,
            "residual_std_mib": None,
        }

    # Fits a polynomial of degree 1, so a line, where times is the
    # x-axis, and cleaned is the y-axis. coefficients[0] is the slope
    # and coefficients[1] is the intercept
    coefficients = np.polyfit(times, cleaned, 1)
    # this takes the line equation and calculates it for each times value.
    # so y = coefficients[0]x + coefficients[1], where x is each times value
    fitted = np.polyval(coefficients, times)

    return {
        "status": "trend_estimated",
        "slope_mib_per_hour": float(coefficients[0] * SECONDS_PER_HOUR),
        "fitted_change_mib": float(coefficients[0] * (times[-1] - times[0])),
        "residual_std_mib": float(np.std(cleaned - fitted)),
    }


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sustained paced streaming on the Raspberry Pi with periodic "
            "CPU, memory, thermal and throttling telemetry."
        )
    )
    parser.add_argument(
        "--precision-label",
        type=str,
        choices=("fp32", "int8"),
        required=True,
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help=(
            "Paced ECG signal time to stream; classifier construction "
            "and record loading are excluded from this budget."
        ),
    )
    parser.add_argument(
        "--monitor-interval-seconds",
        type=float,
        default=DEFAULT_MONITOR_INTERVAL_SECONDS,
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--split-summary-path",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--require-governor",
        type=str,
        default=None,
        help="Exit before running if the observed governor differs.",
    )

    return parser.parse_args()


def _duration_label(duration_seconds: float) -> str:
    if duration_seconds >= 120 and duration_seconds % 60 == 0:
        return f"{int(duration_seconds // 60)}min"

    return f"{int(duration_seconds)}s"


def _raw_arrays(samples: list[dict]) -> dict[str, np.ndarray]:
    def column(key, converter=float):
        return np.asarray(
            [
                converter(sample[key]) if sample[key] is not None else np.nan
                for sample in samples
            ],
            dtype=np.float64,
        )

    return {
        "elapsed_seconds": column("elapsed_seconds"),
        "temperature_c": column("temperature_c"),
        "available_ram_mib": column("available_ram_mib"),
        "rss_mib": column("rss_mib"),
        "system_cpu_percent": column("system_cpu_percent"),
        "process_cpu_percent": column("process_cpu_percent"),
        "cpu_frequency_khz": column("cpu_frequency_khz"),
        "throttled_numeric": column(
            "throttled",
            converter=parse_throttled_flags,
        ),
        "total_chunks_processed": column("total_chunks_processed"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Extract CL arguments
    args = parse_args()

    # Read the governor used by the PI
    governor = read_cpu_governor()

    # Retrieve the expected governor
    expected_governor = args.require_governor or EXPECTED_GOVERNOR

    # If the governor used by the PI is not the expected governor,
    # raise a warning.
    if governor != expected_governor:
        logger.warning(
            "Expected %s governor but observed %s",
            expected_governor,
            governor,
        )

        if args.require_governor is not None:
            raise SystemExit(2)

    # Extract model path
    model_path = args.model_path

    # If no path is provided, create the path based on the input precision
    if model_path is None:
        model_path = (
            DEFAULT_FP32_MODEL_PATH
            if args.precision_label == "fp32"
            else DEFAULT_INT8_MODEL_PATH
        )

    # Load the record names from the validation set
    record_names = load_split_record_names(args.split_summary_path, "val")

    # Cache each record's signal in memory before monitoring begins (~5 MiB each),
    # so disk/network I/O does not interrupt the sustained run. Record transitions
    # then use already-loaded data, keeping telemetry focused on the paced workload.
    prefetched: dict[str, tuple] = {}

    # For each record
    for record_name in record_names:
        # Load the record
        signals, fields, _ = load_record(record_name=record_name)
        # Select the MLII channel
        signal, _ = select_signal_channel(signals=signals, fields=fields)
        # The key becomes the record name, and its value is a tuple that is
        # its signal and its sampling rate
        prefetched[record_name] = (signal, float(fields["fs"]))
        logger.info("Prefetched record %s before monitoring", record_name)

    # Returns the ReplaySource populated by a given record
    def cached_source(record_name: str) -> ReplaySource:
        signal, sampling_rate = prefetched[record_name]

        return ReplaySource(
            signal=signal,
            sampling_rate=sampling_rate,
            chunk_size=args.chunk_size,
            record_name=record_name,
        )

    # Create our classifier
    classifier = ONNXSequenceClassifier(model_path)
    # Create our predictor
    predictor = StreamingPredictor(
        engine=StreamingEngine(),
        classifier=classifier,
    )

    # Warm up the full streaming inference path before we begin measuring.
    # This absorbs one-time ONNX Runtime work without affecting the sustained
    # telemetry, latency or deadline measurements below.
    warmup = warm_up_predictor(
        predictor=predictor,
        source=cached_source(record_names[0]),
    )
    logger.info(
        "Warm-up complete on record %s (%d chunks, %d predictions)",
        warmup["record_name"],
        warmup["chunks_processed"],
        warmup["predictions"],
    )

    # Create progress tracker to track record name and
    # num of chunks through
    progress = ProgressTracker()

    # Create our sampler that repeatably takes telementry samples
    # whilst the ECG inference is happening on the main thread.
    sampler = TelemetrySampler(
        reader=TelemetryReader(),
        progress=progress,
        interval_seconds=args.monitor_interval_seconds,
    )

    logger.info(
        "Sustained %s run for %.0f s over records %s",
        args.precision_label.upper(),
        args.duration_seconds,
        record_names,
    )

    # Starts the telemetry thread, which then executes
    # TelemetrySampler.run() in that new thread.
    sampler.start()

    try:
        # Returns metrics from the streaming run
        streaming = sustained_streaming_run(
            predictor=predictor,
            record_names=record_names,
            source_factory=cached_source,
            duration_ns=round(args.duration_seconds * NANOSECONDS_PER_SECOND),
            chunk_size=args.chunk_size,
            progress=progress,
        )
    finally:
        # Stops the telementry thread from taking telementry samples
        sampler.stop()

    # Returns all the samples returned by the telementry reader during
    # the live infernece run
    samples = sampler.samples
    # returns the time each telemetry sample started
    elapsed = telemetry_column(samples, "elapsed_seconds")

    # Returns a list containing the Pi temperature from each telemetry sample.
    temperature = telemetry_column(samples, "temperature_c")
    # Returns a list of the RSS memory value recorded at each telemetry sample.
    rss = telemetry_column(samples, "rss_mib")
    # Returns a list of how much ram was availble at each telementry sample.
    available = telemetry_column(samples, "available_ram_mib")

    # Returns the start and end RSS values, and calculates the min, max, and mean
    # of the obtained valid RSS values
    rss_summary = series_summary(rss)

    # Capture the trend of the RSS values
    trend = rss_trend(elapsed, rss)

    # returns a throtlling summary, including if we ever observed a throttling
    # state.
    throttling = throttling_summary(
        elapsed,
        telemetry_column(samples, "throttled"),
    )

    result = {
        "environment": environment_metadata(classifier.providers),
        "precision": args.precision_label,
        "model_path": str(model_path),
        "governor_observed": governor,
        "requested_duration_seconds": args.duration_seconds,
        "paced_signal_seconds": streaming["paced_signal_seconds"],
        "actual_duration_seconds": streaming["actual_duration_seconds"],
        "monitor_interval_seconds": args.monitor_interval_seconds,
        "warmup": warmup,
        "telemetry_samples": len(samples),
        "record_cycle": record_names,
        "streaming": {
            "per_record": streaming["per_record"],
            "totals": streaming["totals"],
        },
        "deadline_misses_detail": correlate_misses(
            streaming["misses"],
            samples,
        ),
        "thermal": {
            "summary": series_summary(temperature),
            "time_to_maximum_seconds": time_to_maximum(elapsed, temperature),
            "final_window_mean_c": final_window_mean(elapsed, temperature),
        },
        "cpu": {
            "process_cpu_summary": series_summary(
                telemetry_column(samples, "process_cpu_percent")
            ),
            "system_cpu_summary": series_summary(
                telemetry_column(samples, "system_cpu_percent")
            ),
            "frequency_khz_summary": series_summary(
                telemetry_column(samples, "cpu_frequency_khz")
            ),
        },
        "memory": {
            "rss_summary": rss_summary,
            "rss_trend": trend,
            "available_ram_summary": series_summary(available),
            "available_ram_slope_mib_per_hour": slope_per_hour(
                elapsed,
                available,
            ),
        },
        "throttling": throttling,
        "monitoring": {
            "failure_count": sampler.failure_count,
            "failures": sampler.failures,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.precision_label}_sustained_{_duration_label(args.duration_seconds)}"

    with (args.output_dir / f"{stem}.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    np.savez_compressed(args.output_dir / f"{stem}_raw.npz", **_raw_arrays(samples))

    logger.info("Wrote sustained result to %s", args.output_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
