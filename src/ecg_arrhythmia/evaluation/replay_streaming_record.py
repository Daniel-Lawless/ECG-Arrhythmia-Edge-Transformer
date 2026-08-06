import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from ecg_arrhythmia.streaming.replay_source import (
    DEFAULT_CHUNK_SIZE,
    ReplayMode,
    ReplaySource,
)
from ecg_arrhythmia.streaming.streaming_engine import (
    StreamContinuityError,
    StreamingEngine,
)

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"


@dataclass(frozen=True)
class ReplaySummary:
    """Outcome of replaying one record through the streaming engine."""

    record_name: str
    lead_name: str | None
    replay_mode: str
    sampling_rate: float
    chunk_size: int

    total_input_samples: int
    total_emitted_chunks: int
    total_samples_accepted: int

    first_sample_index: int | None
    final_sample_index: int | None

    continuity_validated: bool
    elapsed_seconds: float

    @property
    def processed_signal_seconds(self) -> float:
        """
        Seconds of ECG actually processed by the engine.

        Based on accepted samples rather than record length, so an early
        stop or a truncated stream is not credited with the whole record.
        """

        return self.total_samples_accepted / self.sampling_rate

    @property
    def real_time_factor(self) -> float | None:
        """
        Processing wall time divided by the ECG duration processed.

        Below one is faster than real time, one is exactly real time and
        above one means the system cannot keep up. ``None`` when no ECG
        was processed. Reported for context only; the full performance
        measurements arrive with the later benchmarking stage.
        """

        signal_seconds = self.processed_signal_seconds

        if signal_seconds <= 0:
            return None

        return self.elapsed_seconds / signal_seconds

    @property
    def speedup_factor(self) -> float | None:
        """
        ECG duration processed divided by processing wall time.

        The reciprocal of the real-time factor, kept because a "600x
        faster than real time" figure is easier to read for accelerated
        replay. ``None`` when no measurable time elapsed.
        """

        if self.elapsed_seconds <= 0:
            return None

        return self.processed_signal_seconds / self.elapsed_seconds


def replay_record(
    source: ReplaySource,
    engine: StreamingEngine,
    max_samples: int | None = None,
) -> ReplaySummary:
    """
    Push every chunk from source through engine and summarise it.

    max_samples stops the replay early, which keeps a real-time run of
    a full 30-minute record practical to sanity check.

    A broken stream is reported rather than hidden. the returned summary
    records continuity_validated=False and the caller decides how to
    fail. main exits non-zero in that case.
    """

    record_name = source.record_name or "unknown"
    # Starts the StreamState for this record.
    engine.start_record(record_name=record_name)

    # Initalise processed chunks to 0 and
    # continuity validity to True.
    emitted_chunks = 0
    continuity_validated = True

    # Records whether every sample in the source has been consumed.
    reached_end_of_record = False

    # Start the timer.
    start_time = perf_counter()

    try:
        # This will deliever one chunk at a time.
        for chunk in source.iter_chunks():
            # Process one chunk. For now this will only check
            # continuity and update the records StreamState.
            engine.process_chunk(chunk)
            emitted_chunks += 1

            if (
                # If we have processed more samples than max_samples,
                # then we have processed the enough seconds of ECG
                # given by limit_seconds, so we stop
                max_samples is not None
                and engine.state.total_samples_accepted >= max_samples
            ):
                # The limit may have landed exactly on the end of the
                # source, in which case this is a genuine record boundary.
                reached_end_of_record = (
                    engine.state.total_samples_accepted >= source.num_samples
                )
                break
        else:
            # The iterator ended naturally, so the whole source was consumed.
            reached_end_of_record = True

        if reached_end_of_record:
            engine.flush()

    # This occurs if the sampling rate changes, or the next chunk
    # does not follow validity.
    except StreamContinuityError:
        continuity_validated = False
        logger.exception("Stream continuity validation failed")

    # Finally runs regardless how the try statement ends.
    # This will tell us how long it took to process the record.
    finally:
        elapsed_seconds = perf_counter() - start_time

    # Load this records StreamState
    state = engine.state

    # Extract summary metrics
    return ReplaySummary(
        record_name=record_name,
        lead_name=source.lead_name,
        replay_mode=str(source.mode),
        sampling_rate=source.sampling_rate,
        chunk_size=source.chunk_size,
        total_input_samples=source.num_samples,
        total_emitted_chunks=emitted_chunks,
        total_samples_accepted=state.total_samples_accepted,
        first_sample_index=state.first_sample_index,
        final_sample_index=state.last_sample_index,
        continuity_validated=continuity_validated,
        elapsed_seconds=elapsed_seconds,
    )


def write_summary(summary: ReplaySummary, output_dir: Path) -> None:
    """
    Write the summary of a record to a file
    """

    # Ensure the out_dir is made
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with output_dir.open("w", encoding="utf-8") as file:
        # We need to use asdict here so it can be serialised
        # by JSON
        json.dump(asdict(summary), file, indent=4)


def parse_args() -> argparse.Namespace:
    # Create our parser
    parser = argparse.ArgumentParser(
        description=("CLI for streaming one record through the streaming engine")
    )

    parser.add_argument("--record-name", type=str, default=DEFAULT_RECORD_NAME)

    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ReplayMode],
        default=ReplayMode.ACCELERATED.value,
    )

    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=None,
        help=(
            "limit on how many seconds of ECG to replay. Useful "
            "for sanity checking real-time mode without waiting for the "
            "whole record."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/streaming_evaluation/streaming_summary.json"),
        help="Where to save streaming summary",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Populates ReplaySource with the record we
    # entered in the CLI
    source = ReplaySource.from_record(
        record_name=args.record_name,
        chunk_size=args.chunk_size,
        mode=args.mode,
    )

    # will process the whole record
    max_samples = None

    if args.limit_seconds is not None:
        # We convert the length of time we want to run for into samples.
        # I.e., if we want to run 2 seconds worth of ECG and the sampling rate
        # is 360, then we would run 2 * 360 = 720 samples.
        max_samples = int(args.limit_seconds * source.sampling_rate)

    # This will push the record through the engine and return
    # a summary
    summary = replay_record(
        source=source,
        engine=StreamingEngine(),
        max_samples=max_samples,
    )

    # Write the summary to a file
    write_summary(summary=summary, output_dir=args.output_dir)

    # A stream that failed continuity validation must not look like a
    # successful run to whatever invoked this command.
    if not summary.continuity_validated:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
