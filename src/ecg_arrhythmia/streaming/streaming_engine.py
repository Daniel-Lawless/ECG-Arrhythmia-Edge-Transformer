import logging
from dataclasses import dataclass

from ecg_arrhythmia.streaming.sample_chunk import SampleChunk, validate_start_index

logger = logging.getLogger(__name__)


class StreamContinuityError(ValueError):
    """Raised when an incoming chunk does not continue the current stream."""


@dataclass
class StreamState:
    """
    A StreamState object records the progress through
    the processing of a record. It is mutable to allow us to
    update the state as we are processing.

    The state is scoped to one record. Starting a new record resets it so
    samples from one record can never influence another.
    """

    record_name: str | None = None
    sampling_rate: float | None = None

    # Absolute index the next accepted chunk must start at.
    next_expected_index: int = 0

    total_samples_accepted: int = 0
    num_chunks_accepted: int = 0

    # Absolute index of the first accepted sample, for reporting.
    first_sample_index: int | None = None

    @property
    def last_sample_index(self) -> int | None:
        """Absolute index of the most recently accepted sample."""

        if self.num_chunks_accepted == 0:
            return None

        return self.next_expected_index - 1


class StreamingEngine:
    """
    Stateful consumer of consecutive ECG sample chunks.

    In this first streaming stage the engine only guarantees that a record
    arrives as one unbroken, correctly ordered sample stream at a constant
    sampling rate. Detection, feature extraction and inference are added
    in later stages and will attach to the same process_chunk call.

    The engine is deliberately independent of where the chunks come from,
    so a replay source, a Raspberry Pi, or a future device
    driver can all feed it the same SampleChunk objects.
    """

    def __init__(self) -> None:
        self._state = StreamState()

    @property
    def state(self) -> StreamState:
        """Current stream state, scoped to the record being processed."""

        return self._state

    def start_record(
        self,
        record_name: str | None = None,
        start_index: int = 0,
    ) -> None:
        """
        Begin a new record, discarding any previous stream state.

        Record boundaries are explicit so no counters, sampling rate or
        expected sample position can leak between records.

        start_index is the Absolute index the first chunk of this
        record must start at. Defaults to zero. We supply it when a
        streaming deliberately begins part-way through a record,
        so callers never have to modify continuity state by hand.
        """

        self._state = StreamState(
            record_name=record_name,
            next_expected_index=validate_start_index(start_index),
        )
        logger.debug(
            "Started streaming record %s at sample %s",
            record_name,
            start_index,
        )

    def reset(self) -> None:
        """Discard all stream state without naming a new record."""

        self.start_record(record_name=None, start_index=0)

    def process_chunk(self, chunk: SampleChunk) -> None:
        """
        Accept the next chunk of the current stream.

        The chunk must start exactly where the previous one ended and must
        carry the same sampling rate as the rest of the record. Gaps,
        overlaps, duplicates and out-of-order chunks are rejected rather
        than silently corrupting the stream.

        Later stages will return the events produced by this chunk, which
        callers can adopt without changing existing call sites.
        """

        # Validates if this chunks sampling rate matches
        # the StreamState sampling rate for the record it is in
        self._validate_sampling_rate(chunk)

        # Checks that this next chunk is valid to run. I.e.,
        # it is not a duplicate, starts in the right place,
        # does not overlap, etc.
        self._validate_continuity(chunk)

        # If we have not seen a chunk yet, make its start
        # index equal to the start index of the first chunk
        if self._state.first_sample_index is None:
            self._state.first_sample_index = chunk.start_index

        # Update this records state.
        self._state.sampling_rate = chunk.sampling_rate
        self._state.next_expected_index = chunk.stop_index
        self._state.total_samples_accepted += chunk.num_samples
        self._state.num_chunks_accepted += 1

    def _validate_sampling_rate(self, chunk: SampleChunk) -> None:
        """Reject a sampling-rate change part-way through a record."""

        # This is the sampling rate of this record.
        stream_sampling_rate = self._state.sampling_rate

        # If the chunks sampling rate differs to the sampling rate
        # of this record, it should be rejected.
        if (
            stream_sampling_rate is not None
            and chunk.sampling_rate != stream_sampling_rate
        ):
            raise StreamContinuityError(
                "Sampling rate changed mid-stream: expected "
                f"{stream_sampling_rate} Hz but received "
                f"{chunk.sampling_rate} Hz."
            )

    def _validate_continuity(self, chunk: SampleChunk) -> None:
        """Reject any chunk that does not continue the stream exactly."""

        # This is the index of the next chunk the record expects
        expected_index = self._state.next_expected_index

        # If this next chunk does start at the expected index,
        # then it is safe to continue
        if chunk.start_index == expected_index:
            return

        # If the chunk index start after the expected index, then some
        # samples have been skipped, so a continuity error should be thrown.
        if chunk.start_index > expected_index:
            missing_samples = chunk.start_index - expected_index
            raise StreamContinuityError(
                f"Gap of {missing_samples} samples: expected a chunk "
                f"starting at {expected_index} but received one starting "
                f"at {chunk.start_index}."
            )

        # If the stop index of the next chunk is equal to the expected index,
        # then this is the chunk that we just processed, so it has been
        # duplicated and thus should be rejected.
        if chunk.stop_index == expected_index:
            raise StreamContinuityError(
                f"Duplicate chunk covering samples {chunk.start_index} to "
                f"{chunk.last_index}: the stream already reached "
                f"{expected_index}."
            )

        # A chunk starting before the expected position is either the
        # previous chunk resent, or an overlapping/out-of-order chunk.
        raise StreamContinuityError(
            "Overlapping or out-of-order chunk: expected a chunk starting "
            f"at {expected_index} but received one starting at "
            f"{chunk.start_index}."
        )
