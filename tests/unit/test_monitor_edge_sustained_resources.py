import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.evaluation.monitor_edge_sustained_resources import (
    ProgressTracker,
    TelemetryReader,
    TelemetrySampler,
    correlate_misses,
    final_window_mean,
    rss_trend,
    series_summary,
    slope_per_hour,
    sustained_streaming_run,
    throttling_summary,
    time_to_maximum,
    warm_up_predictor,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import PredictionEvent
from ecg_arrhythmia.telemetry.edge_sensors import (
    parse_proc_stat,
    parse_process_jiffies,
    parse_thermal_zone_temp,
    parse_vmrss_mib,
    process_cpu_percent,
    read_temperature_c,
    read_vmrss_mib,
    system_cpu_percent,
)

MS = 1_000_000  # nanoseconds per millisecond


# ---------------------------------------------------------------------
#                        Linux Telemetry Parsers
# ---------------------------------------------------------------------


def test_vmrss_is_parsed_into_mib():
    status = "VmPeak:\t 300000 kB\nVmRSS:\t  204800 kB\nVmData:\t 1 kB\n"

    assert parse_vmrss_mib(status) == pytest.approx(200.0)


def test_missing_or_malformed_vmrss_degrades_to_none(tmp_path):
    assert parse_vmrss_mib("VmPeak: 1 kB\n") is None
    assert parse_vmrss_mib("VmRSS:\tgarbage kB\n") is None
    assert read_vmrss_mib(tmp_path / "missing") is None


def test_proc_stat_busy_and_total_jiffies_are_extracted():
    stat = "cpu  10 0 10 70 10 0 0 0 0 0\ncpu0 1 2 3 4 5 6 7 8 9 0\n"

    busy, total = parse_proc_stat(stat)

    # Idle (70) + iowait (10) are excluded from busy time.
    assert busy == 20
    assert total == 100


def test_malformed_proc_stat_degrades_to_none():
    assert parse_proc_stat("intr 12345\n") is None
    assert parse_proc_stat("cpu  ten zero\n") is None


def test_system_cpu_percent_from_successive_counters():
    assert system_cpu_percent((20, 100), (50, 200)) == pytest.approx(30.0)
    # Fully idle interval.
    assert system_cpu_percent((20, 100), (20, 200)) == pytest.approx(0.0)
    # Fully busy interval.
    assert system_cpu_percent((20, 100), (120, 200)) == pytest.approx(100.0)


def test_system_cpu_percent_degrades_without_two_readings():
    assert system_cpu_percent(None, (50, 200)) is None
    assert system_cpu_percent((20, 100), None) is None
    assert system_cpu_percent((20, 100), (20, 100)) is None


def test_process_cpu_percent_is_percent_of_one_core():
    # 50 jiffies at 100 ticks/s over one second: half of one core.
    assert process_cpu_percent(50, 1.0, 100) == pytest.approx(50.0)
    # All four Pi cores busy: the documented convention exceeds 100%.
    assert process_cpu_percent(400, 1.0, 100) == pytest.approx(400.0)
    assert process_cpu_percent(50, 0.0, 100) is None


def test_process_jiffies_survive_a_comm_field_with_spaces_and_parens():
    stat = (
        "1234 (my (weird) proc) S 1 1 1 0 -1 4194560 500 0 0 0 "
        "150 50 0 0 20 0 4 0 100 1000000 250"
    )

    assert parse_process_jiffies(stat) == 200


def test_malformed_process_stat_degrades_to_none():
    assert parse_process_jiffies("no closing paren") is None
    assert parse_process_jiffies("1 (x) S 1 2") is None


def test_thermal_zone_millidegrees_are_parsed():
    assert parse_thermal_zone_temp("48237\n") == pytest.approx(48.237)
    assert parse_thermal_zone_temp("garbage") is None


def test_temperature_falls_back_to_vcgencmd(tmp_path):
    temperature = read_temperature_c(
        thermal_path=tmp_path / "missing",
        vcgencmd=lambda argument: "temp=50.1'C",
    )

    assert temperature == pytest.approx(50.1)


# ---------------------------------------------------------------------
#                        Telemetry Sampling
# ---------------------------------------------------------------------


def _write_proc_files(tmp_path, busy, total, jiffies, rss_kib):
    idle = total - busy
    (tmp_path / "stat").write_text(f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\n")
    (tmp_path / "self_stat").write_text(
        f"1 (proc) S 1 1 1 0 -1 0 0 0 0 0 {jiffies} 0 0 0 20 0 4 0 1 1 1"
    )
    (tmp_path / "self_status").write_text(f"VmRSS:\t{rss_kib} kB\n")
    (tmp_path / "meminfo").write_text("MemTotal: 1013712 kB\nMemAvailable: 665600 kB\n")
    (tmp_path / "temp").write_text("48237\n")


def _reader(tmp_path) -> TelemetryReader:
    return TelemetryReader(
        meminfo_path=tmp_path / "meminfo",
        proc_stat_path=tmp_path / "stat",
        process_stat_path=tmp_path / "self_stat",
        process_status_path=tmp_path / "self_status",
        thermal_path=tmp_path / "temp",
        vcgencmd=lambda argument: "throttled=0x0",
        frequency_reader=lambda: 2400000,
        governor_reader=lambda: "performance",
        ticks_per_second=100,
    )


def test_the_first_sample_has_fields_but_no_cpu_percentages(tmp_path):
    _write_proc_files(tmp_path, busy=20, total=100, jiffies=10, rss_kib=204800)
    reader = _reader(tmp_path)

    sample = reader.sample(0.0, ProgressTracker())

    assert sample["temperature_c"] == pytest.approx(48.237)
    assert sample["rss_mib"] == pytest.approx(200.0)
    assert sample["available_ram_mib"] == pytest.approx(650.0)
    assert sample["cpu_frequency_khz"] == 2400000
    assert sample["cpu_governor"] == "performance"
    assert sample["throttled"] == "0x0"
    assert sample["system_cpu_percent"] is None
    assert sample["process_cpu_percent"] is None


def test_the_second_sample_computes_exact_cpu_percentages(tmp_path):
    _write_proc_files(tmp_path, busy=20, total=100, jiffies=10, rss_kib=204800)
    reader = _reader(tmp_path)
    progress = ProgressTracker()
    progress.record_name = "114"
    progress.total_chunks = 42

    reader.sample(0.0, progress)

    # One second later: 30 more busy jiffies of 100 total, and the
    # process consumed 50 jiffies at 100 ticks per second.
    _write_proc_files(tmp_path, busy=50, total=200, jiffies=60, rss_kib=204800)
    sample = reader.sample(1.0, progress)

    assert sample["system_cpu_percent"] == pytest.approx(30.0)
    assert sample["process_cpu_percent"] == pytest.approx(50.0)
    assert sample["record_name"] == "114"
    assert sample["total_chunks_processed"] == 42


class FakeStopEvent:
    """Stop event whose wait() outcomes are scripted, with no sleeping."""

    def __init__(self, waits: list[bool]) -> None:
        self.waits = waits

    def wait(self, timeout: float) -> bool:
        return self.waits.pop(0)

    def set(self) -> None:  # pragma: no cover - interface compatibility
        pass


class CountingClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        self.now_ns += 5 * 1_000_000_000

        return self.now_ns


def test_the_sampler_collects_until_stopped_without_sleeping(tmp_path):
    _write_proc_files(tmp_path, busy=20, total=100, jiffies=10, rss_kib=1024)
    sampler = TelemetrySampler(
        reader=_reader(tmp_path),
        progress=ProgressTracker(),
        interval_seconds=5.0,
        clock=CountingClock(),
        stop_event=FakeStopEvent([False, False, True]),
    )

    sampler.run()

    assert len(sampler.samples) == 3
    assert sampler.failure_count == 0
    # Elapsed values advance with the fake clock.
    elapsed = [sample["elapsed_seconds"] for sample in sampler.samples]
    assert elapsed == sorted(elapsed)


def test_sampler_failures_are_counted_not_fatal():
    class ExplodingReader:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self, elapsed, progress):
            self.calls += 1

            if self.calls == 2:
                raise OSError("sensor unavailable")

            return {"elapsed_seconds": elapsed}

    sampler = TelemetrySampler(
        reader=ExplodingReader(),
        progress=ProgressTracker(),
        interval_seconds=5.0,
        clock=CountingClock(),
        stop_event=FakeStopEvent([False, False, True]),
    )

    sampler.run()

    assert len(sampler.samples) == 2
    assert sampler.failure_count == 1
    assert "sensor unavailable" in sampler.failures[0]


# ---------------------------------------------------------------------
#                       Telemetry Analysis
# ---------------------------------------------------------------------


def test_a_linear_rss_series_recovers_its_slope():
    elapsed = [0.0, 1800.0, 3600.0]
    values = [100.0, 105.0, 110.0]

    assert slope_per_hour(elapsed, values) == pytest.approx(10.0, rel=1e-6)


def test_a_flat_series_has_a_near_zero_slope_and_nones_are_skipped():
    elapsed = [0.0, 10.0, 20.0, 30.0]
    values = [50.0, None, 50.0, 50.0]

    assert slope_per_hour(elapsed, values) == pytest.approx(0.0, abs=1e-9)
    assert slope_per_hour([0.0, 10.0], [None, None]) is None
    assert slope_per_hour([0.0], [5.0]) is None


def test_series_summary_reports_endpoints_and_extremes():
    summary = series_summary([None, 10.0, 20.0, 15.0])

    assert summary["start"] == pytest.approx(10.0)
    assert summary["end"] == pytest.approx(15.0)
    assert summary["minimum"] == pytest.approx(10.0)
    assert summary["maximum"] == pytest.approx(20.0)
    assert summary["mean"] == pytest.approx(15.0)
    assert summary["delta"] == pytest.approx(5.0)
    assert series_summary([None, None]) is None


def test_final_window_mean_uses_only_the_trailing_window():
    elapsed = [0.0, 300.0, 600.0, 900.0, 1200.0]
    values = [40.0, 50.0, 60.0, 70.0, 70.0]

    mean = final_window_mean(elapsed, values, window_seconds=600.0)

    assert mean == pytest.approx((60.0 + 70.0 + 70.0) / 3)


def test_time_to_maximum_finds_the_peak_sample():
    assert time_to_maximum(
        [0.0, 5.0, 10.0],
        [40.0, 70.0, 60.0],
    ) == pytest.approx(5.0)


def test_clean_throttling_is_summarised_as_clean():
    summary = throttling_summary([0.0, 5.0], ["0x0", "0x0"])

    assert not summary["any_throttling_observed"]
    assert summary["first_throttling_elapsed_seconds"] is None
    assert summary["unique_values"] == ["0x0"]
    assert summary["final_value"] == "0x0"


def test_observed_throttling_is_surfaced_with_its_first_timestamp():
    summary = throttling_summary(
        [0.0, 5.0, 10.0, 15.0],
        ["0x0", None, "0x50000", "0x50005"],
    )

    assert summary["any_throttling_observed"]
    assert summary["first_throttling_elapsed_seconds"] == pytest.approx(10.0)
    assert summary["unique_values"] == ["0x0", "0x50000", "0x50005"]
    assert summary["samples_with_readings"] == 3


def test_misses_are_correlated_with_the_nearest_telemetry_sample():
    samples = [
        {
            "elapsed_seconds": t,
            "temperature_c": 40.0 + t,
            "cpu_frequency_khz": 2400000,
            "rss_mib": 200.0,
            "available_ram_mib": 650.0,
        }
        for t in (0.0, 5.0, 10.0)
    ]
    misses = [{"elapsed_seconds": 6.2, "deadline_lateness_ms": 3.0}]

    correlated = correlate_misses(misses, samples)

    nearest = correlated[0]["nearest_telemetry"]
    assert nearest["elapsed_seconds"] == pytest.approx(5.0)
    assert nearest["temperature_c"] == pytest.approx(45.0)


# ---------------------------------------------------------------------
#                   Sustained Multi-Record Lifecycle
# ---------------------------------------------------------------------


class FakeTime:
    def __init__(self) -> None:
        self.now_ns = 0

    def clock(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += round(seconds * 1_000_000_000)


def make_event(target_peak: int) -> PredictionEvent:
    logits = np.zeros(4, dtype=np.float32)
    logits[0] = 1.0

    return PredictionEvent(
        target_peak_index=target_peak,
        peak_indices=(target_peak,),
        logits=logits,
        predicted_class_index=0,
        predicted_label="N",
    )


class FakeSustainedPredictor:
    """Instant predictor that logs record boundaries and flushes."""

    def __init__(self) -> None:
        self.started_records: list[str] = []
        self.flush_calls = 0
        self.engine = types.SimpleNamespace(
            state=types.SimpleNamespace(total_samples_accepted=0)
        )

    def start_record(self, record_name: str) -> None:
        self.started_records.append(record_name)
        # StreamingEngine state is scoped to one record, so mirror that
        # behaviour in the fake used by the sustained-run tests.
        self.engine.state.total_samples_accepted = 0

    def process_chunk(self, chunk) -> list[PredictionEvent]:
        # Mirror the real StreamingEngine's sample accounting so the sustained
        # duration budget advances by the amount of signal processed.
        self.engine.state.total_samples_accepted += chunk.num_samples
        return []

    def flush(self) -> list[PredictionEvent]:
        self.flush_calls += 1

        return [make_event(100 * self.flush_calls)]


class FakeSource:
    def __init__(
        self,
        record_name: str,
        num_chunks: int | None = None,
        chunk_samples: list[int] | None = None,
    ) -> None:
        if chunk_samples is None:
            chunk_samples = [36] * (num_chunks or 0)

        self.record_name = record_name
        self.chunk_samples = chunk_samples
        self.num_chunks = len(chunk_samples)
        self.num_samples = sum(chunk_samples)
        self.sampling_rate = 360.0

    def iter_chunks(self):
        for samples in self.chunk_samples:
            yield types.SimpleNamespace(num_samples=samples)


class FakeWarmupPredictor:
    """Predictor that emits one prediction on its second streamed chunk."""

    def __init__(self) -> None:
        self.started_records: list[str] = []
        self.process_calls = 0
        self.flush_calls = 0
        self.reset_calls = 0

    def start_record(self, record_name: str) -> None:
        self.started_records.append(record_name)

    def process_chunk(self, chunk) -> list[PredictionEvent]:
        self.process_calls += 1
        return [make_event(100)] if self.process_calls == 2 else []

    def flush(self) -> list[PredictionEvent]:
        self.flush_calls += 1
        return []

    def reset(self) -> None:
        self.reset_calls += 1


def test_warmup_stops_after_first_prediction_and_resets():
    predictor = FakeWarmupPredictor()
    source = FakeSource("A", num_chunks=4)

    result = warm_up_predictor(predictor, source)

    assert predictor.started_records == ["A_warmup"]
    assert predictor.process_calls == 2
    assert predictor.flush_calls == 0
    assert predictor.reset_calls == 1
    assert result == {
        "record_name": "A",
        "chunks_processed": 2,
        "predictions": 1,
    }


def test_records_cycle_with_clean_boundaries_and_truncation():
    fake_time = FakeTime()
    predictor = FakeSustainedPredictor()
    progress = ProgressTracker()

    result = sustained_streaming_run(
        predictor=predictor,
        record_names=["A", "B"],
        source_factory=lambda name: FakeSource(name, num_chunks=3),
        duration_ns=500 * MS,
        chunk_size=36,
        progress=progress,
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
    )

    per_record = result["per_record"]

    # The budget is paced signal time: A streams 300 ms of signal
    # (3 chunks), leaving 200 ms, so B is cut to a two-chunk budget.
    assert [record["record_name"] for record in per_record] == ["A", "B"]
    assert [record["truncated"] for record in per_record] == [False, True]
    assert [record["chunks_processed"] for record in per_record] == [3, 2]

    # Every record got its own start_record and flush: no state leaks.
    assert predictor.started_records == ["A", "B"]
    assert predictor.flush_calls == 2

    assert result["totals"]["chunks"] == 5
    assert result["totals"]["predictions"] == 2
    assert result["totals"]["deadline_misses"] == 0
    assert result["paced_signal_seconds"] == pytest.approx(0.5)
    assert progress.total_chunks == 5


def test_signal_time_counts_samples_not_nominal_chunk_periods():
    # A record whose final chunk holds 18 samples contributes 54/360 =
    # 0.15 s of signal, not 2 x 100 ms of nominal chunk period.
    fake_time = FakeTime()
    predictor = FakeSustainedPredictor()

    result = sustained_streaming_run(
        predictor=predictor,
        record_names=["A"],
        source_factory=lambda name: FakeSource(name, chunk_samples=[36, 18]),
        duration_ns=200 * MS,
        chunk_size=36,
        progress=ProgressTracker(),
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
    )

    assert [record["chunks_processed"] for record in result["per_record"]] == [2]
    assert result["per_record"][0]["truncated"] is False
    assert result["paced_signal_seconds"] == pytest.approx(0.15)


def test_the_run_stops_before_starting_an_unaffordable_record():
    fake_time = FakeTime()
    predictor = FakeSustainedPredictor()

    result = sustained_streaming_run(
        predictor=predictor,
        record_names=["A"],
        source_factory=lambda name: FakeSource(name, num_chunks=2),
        duration_ns=50 * MS,  # Less than one chunk period.
        chunk_size=36,
        progress=ProgressTracker(),
        clock=fake_time.clock,
        sleeper=fake_time.sleep,
    )

    assert result["per_record"] == []
    assert predictor.started_records == []


# ---------------------------------------------------------------------
#                           RSS Trend
# ---------------------------------------------------------------------


def test_a_linear_trend_is_estimated_without_a_verdict():
    # A clean 10 MiB/hour ramp: exact numbers, no growth decision.
    elapsed = [0.0, 900.0, 1800.0, 2700.0, 3600.0]
    values = [100.0, 102.5, 105.0, 107.5, 110.0]

    trend = rss_trend(elapsed, values)

    assert trend["status"] == "trend_estimated"
    assert trend["slope_mib_per_hour"] == pytest.approx(10.0, rel=1e-6)
    assert trend["fitted_change_mib"] == pytest.approx(10.0, rel=1e-6)
    assert trend["residual_std_mib"] == pytest.approx(0.0, abs=1e-9)


def test_noisy_telemetry_reports_its_scatter_alongside_the_slope():
    elapsed = [0.0, 900.0, 1800.0, 2700.0, 3600.0]
    values = [200.0, 204.0, 197.0, 203.0, 199.0]

    trend = rss_trend(elapsed, values)

    assert trend["status"] == "trend_estimated"
    assert trend["slope_mib_per_hour"] == pytest.approx(-1.2, rel=1e-6)
    assert trend["residual_std_mib"] > 0


def test_too_few_points_give_an_insufficient_data_trend():
    trend = rss_trend([0.0, 5.0], [100.0, 101.0])

    assert trend["status"] == "insufficient_data"
    assert trend["slope_mib_per_hour"] is None


# ---------------------------------------------------------------------
#                        Runtime-Light Import
# ---------------------------------------------------------------------


def test_importing_the_sustained_monitor_loads_neither_torch_nor_matplotlib():
    script = (
        "import sys\n"
        "import ecg_arrhythmia.evaluation.monitor_edge_sustained_resources\n"
        "blocked = [name for name in sys.modules "
        "if name == 'torch' or name.startswith('torch.') "
        "or name == 'matplotlib' or name.startswith('matplotlib.')]\n"
        "assert not blocked, f'forbidden modules imported: {blocked}'\n"
        "print('runtime-light import OK')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 0, completed.stderr
    assert "runtime-light import OK" in completed.stdout
