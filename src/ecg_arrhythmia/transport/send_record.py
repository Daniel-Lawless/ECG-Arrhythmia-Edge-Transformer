import argparse
import logging
import time
from pathlib import Path

from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier
from ecg_arrhythmia.streaming.replay_source import (
    DEFAULT_CHUNK_SIZE,
    ReplayMode,
    ReplaySource,
)
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.streaming.streaming_predictor import StreamingPredictor
from ecg_arrhythmia.telemetry.live import LiveEdgeTelemetry
from ecg_arrhythmia.transport.tcp_receiver import DEFAULT_PORT
from ecg_arrhythmia.transport.tcp_sender import TCPStreamSender

logger = logging.getLogger(__name__)

# The Section 5.5 deployment default precision.
DEFAULT_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_RECORD_NAME = "114"

DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS = 1.0

NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1000.0


class IntervalTimingAccumulator:
    """
    Worst-case process_chunk() duration in the current telemetry interval.

    The workload is bursty: most chunks cost microseconds while the
    occasional detector-stride chunk runs the model, so the latest
    duration alone would usually show a meaningless tiny value. The
    interval maximum is what the dashboard should display. reset() is
    called after each runtime_status emission.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.latest_ms: float | None = None
        self.window_max_ms: float | None = None

    def record(self, duration_ms: float) -> None:
        self.latest_ms = duration_ms

        if self.window_max_ms is None or duration_ms > self.window_max_ms:
            self.window_max_ms = duration_ms


class ModelTimingAccumulator:
    """
    Timed model-stage inference work in the current telemetry interval.

    Records the duration of each classifier predict() call and the
    number of sequences it processed (one per call under the current
    classifier contract, but counted rather than assumed). From these:

        mean latency  = total timed ns / calls
        throughput    = sequences / total timed seconds

    Throughput is model-stage sequence capacity, never the ECG
    prediction or heartbeat rate. Both derivations return None for an
    empty interval - an unmeasured value is null, never zero. reset()
    is called after each runtime_status captures the interval.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.sequences = 0
        self.total_ns = 0

    def record(self, elapsed_ns: int, sequences: int = 1) -> None:
        self.calls += 1
        self.sequences += sequences
        self.total_ns += elapsed_ns

    @property
    def mean_latency_ms(self) -> float | None:
        if self.calls == 0:
            return None

        return self.total_ns / self.calls / NANOSECONDS_PER_MILLISECOND

    @property
    def throughput_sequences_per_second(self) -> float | None:
        if self.sequences == 0 or self.total_ns <= 0:
            return None

        return self.sequences / (self.total_ns / NANOSECONDS_PER_SECOND)


class TimedClassifier:
    """
    Composition wrapper timing exactly the classifier predict() call.

    This is the same model-stage seam Section 5.2's benchmark times
    (time_inference in benchmark_onnx_inference): input preparation,
    session.run and logits validation, nothing else. The wrapped
    classifier is completely unchanged - inputs, outputs, batching and
    execution configuration are untouched, and the inner predict()
    result is returned exactly. Accumulator bookkeeping runs after the
    measured region so it never inflates the measurement.
    """

    def __init__(
        self,
        classifier,
        accumulator: ModelTimingAccumulator,
        timer=time.perf_counter_ns,
    ) -> None:
        self._classifier = classifier
        self._accumulator = accumulator
        self._timer = timer

    def predict(self, sequence):
        started = self._timer()
        event = self._classifier.predict(sequence)
        elapsed_ns = self._timer() - started

        self._accumulator.record(elapsed_ns)

        return event


def nominal_chunk_period_ms(source) -> float:
    """The chunk period implied by the source's own configuration."""

    return source.chunk_size / source.sampling_rate * MILLISECONDS_PER_SECOND


def stream_record_to_sender(
    sender,
    source,
    predictor,
    telemetry=None,
    model_timing: ModelTimingAccumulator | None = None,
    status_interval_seconds: float = DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
    should_stop=None,
    clock=time.monotonic,
    timer=time.perf_counter_ns,
) -> dict:
    """
    Drive the pipeline and transport in the documented message order.

    For every chunk: the sample_chunk frame first, then any predictions
    that chunk completed, then one runtime_status if telemetry is
    enabled and its interval is due; flush predictions after all
    chunks. Only predictor.process_chunk() is timed - sends, telemetry
    sampling and pacing are outside the timer. When model_timing is
    provided (fed by a TimedClassifier inside the predictor), each
    runtime_status carries the latest interval's model-stage mean
    latency and throughput; quiet intervals retain the most recent
    valid measurement, dated by its age, and both fields are null
    before any inference. Returns send counts for reporting. Every
    collaborator is a plain parameter so this orchestration is
    testable with fakes and injected clocks.

    should_stop, when given, is polled once per chunk BEFORE that
    chunk is sent; returning True ends the replay early at a chunk
    boundary and still flushes any pending predictions, so the stream
    always stops on a coherent message sequence rather than mid-record.
    The summary reports stopped_early. Default None leaves the loop
    byte-for-byte equivalent to the uninterruptible original.
    """

    record_name = source.record_name
    predictor.start_record(record_name=record_name)

    chunks_sent = 0
    predictions_sent = 0
    runtime_statuses_sent = 0

    accumulator = IntervalTimingAccumulator()
    period_ms = nominal_chunk_period_ms(source) if telemetry is not None else None
    next_status_time = (
        clock() + status_interval_seconds if telemetry is not None else None
    )
    latest_sample_index = None

    # Retained model-stage measurement: quiet intervals re-send the
    # most recent valid values (with a growing age) instead of
    # flashing the dashboard back to unavailable. None until the
    # first-ever timed inference.
    latest_model_mean_ms: float | None = None
    latest_model_throughput: float | None = None
    last_model_measured_at: float | None = None

    stopped_early = False

    for chunk in source.iter_chunks():
        if should_stop is not None and should_stop():
            # Leave the loop at a chunk boundary: predictions already
            # produced are still flushed below.
            stopped_early = True

            break

        sender.send_sample_chunk(chunk, record_name=record_name)
        chunks_sent += 1

        started = timer()
        events = predictor.process_chunk(chunk)
        accumulator.record((timer() - started) / NANOSECONDS_PER_MILLISECOND)
        latest_sample_index = chunk.last_index

        for event in events:
            sender.send_prediction(event, record_name=record_name)
            predictions_sent += 1

        if (
            telemetry is not None
            and clock() >= next_status_time
            and accumulator.window_max_ms is not None
        ):
            # Sample hardware telemetry outside the processing timer.
            hardware = telemetry.sample()

            # Capture this interval's model-stage measurement if any
            # inference ran; otherwise retain the previous one. The
            # accumulator resets only after a capture, so a partial
            # interval is never silently discarded.
            if model_timing is not None and model_timing.calls > 0:
                latest_model_mean_ms = model_timing.mean_latency_ms
                latest_model_throughput = model_timing.throughput_sequences_per_second
                last_model_measured_at = clock()
                model_timing.reset()

            model_age = (
                clock() - last_model_measured_at
                if last_model_measured_at is not None
                else None
            )

            sender.send_runtime_status(
                {
                    "record_name": record_name,
                    "latest_sample_index": latest_sample_index,
                    **hardware,
                    "window_max_chunk_processing_ms": accumulator.window_max_ms,
                    "window_min_processing_headroom_ms": (
                        period_ms - accumulator.window_max_ms
                    ),
                    "model_inference_mean_ms": latest_model_mean_ms,
                    "model_throughput_sequences_per_second": (latest_model_throughput),
                    "model_measurement_age_seconds": model_age,
                }
            )
            runtime_statuses_sent += 1
            accumulator.reset()
            next_status_time = clock() + status_interval_seconds

    flush_predictions_sent = 0

    for event in predictor.flush():
        sender.send_prediction(event, record_name=record_name)
        flush_predictions_sent += 1

    return {
        "chunks_sent": chunks_sent,
        "predictions_sent": predictions_sent,
        "flush_predictions_sent": flush_predictions_sent,
        "runtime_statuses_sent": runtime_statuses_sent,
        "stopped_early": stopped_early,
    }


def run_record_stream(
    host: str,
    port: int = DEFAULT_PORT,
    record: str = DEFAULT_RECORD_NAME,
    model_path: Path = DEFAULT_MODEL_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    mode: str = ReplayMode.REAL_TIME.value,
    runtime_status_interval: float = DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
    should_stop=None,
) -> dict:
    """
    Build the pipeline and stream one record to a receiver.

    The reusable entry point beneath the CLI: main() parses arguments
    and calls this, and the Section 6.5 control agent calls it directly
    on a worker thread. Constructing the source, model session,
    predictor and telemetry here (rather than in main) keeps every
    caller on one identical code path, so a dashboard-started stream is
    the same pipeline the Section 5 benchmarks measured.

    A fresh ONNX session is created per run, deliberately: reusing one
    across records would diverge from the benchmarked configuration for
    the sake of a few hundred milliseconds at startup.

    should_stop is forwarded to stream_record_to_sender, allowing a
    caller to end the replay early at a chunk boundary.
    """

    source = ReplaySource.from_record(
        record_name=record,
        chunk_size=chunk_size,
        mode=mode,
    )
    model_timing = ModelTimingAccumulator()
    predictor = StreamingPredictor(
        engine=StreamingEngine(),
        classifier=TimedClassifier(
            ONNXSequenceClassifier(model_path),
            model_timing,
        ),
    )

    telemetry = LiveEdgeTelemetry() if runtime_status_interval > 0 else None

    with TCPStreamSender(host=host, port=port) as sender:
        return stream_record_to_sender(
            sender,
            source,
            predictor,
            telemetry=telemetry,
            model_timing=model_timing,
            status_interval_seconds=runtime_status_interval,
            should_stop=should_stop,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Stream one record through the FP32 pipeline to a PC receiver "
            "(Section 6.1 demo sender)."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Receiver address (for example the PC's Ethernet IP).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--record", type=str, default=DEFAULT_RECORD_NAME)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in ReplayMode],
        default=ReplayMode.REAL_TIME.value,
        help="real_time paces the stream like a live device (default).",
    )
    parser.add_argument(
        "--runtime-status-interval",
        type=float,
        default=DEFAULT_RUNTIME_STATUS_INTERVAL_SECONDS,
        help=(
            "Seconds between runtime_status messages; a value <= 0 "
            "disables runtime telemetry."
        ),
    )

    args = parser.parse_args()

    summary = run_record_stream(
        host=args.host,
        port=args.port,
        record=args.record,
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        mode=args.mode,
        runtime_status_interval=args.runtime_status_interval,
    )

    logger.info(
        "Streamed record %s: %d chunks, %d predictions, %d flush "
        "predictions, %d runtime statuses",
        args.record,
        summary["chunks_sent"],
        summary["predictions_sent"],
        summary["flush_predictions_sent"],
        summary["runtime_statuses_sent"],
    )


if __name__ == "__main__":
    main()
