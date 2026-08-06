import logging
from dataclasses import dataclass

from ecg_arrhythmia.detection.r_peak_detector import RPeakDetector
from ecg_arrhythmia.preprocessing.beat_extraction import (
    SAMPLING_RATE as MODEL_SAMPLING_RATE,
)
from ecg_arrhythmia.preprocessing.beat_extraction import SEQUENCE_LENGTH
from ecg_arrhythmia.streaming.sample_buffer import IndexedSampleBuffer
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk, validate_start_index
from ecg_arrhythmia.streaming.sequence_assembler import (
    BeatSequence,
    SequenceAssembler,
)
from ecg_arrhythmia.streaming.streaming_detector import (
    DetectorTiming,
    StreamingXQRS,
)

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

    The engine guarantees that a record arrives as one unbroken, correctly
    ordered sample stream at a constant sampling rate, then converts that
    stream causally into model-ready beat sequences. It owns three
    collaborators and no processing logic of its own: a rolling indexed
    buffer, a causal R-peak detector and a sequence assembler.

    The engine is deliberately independent of where the chunks come from,
    so a replay source, a Raspberry Pi, or a future device driver can all
    feed it the same SampleChunk objects.
    """

    def __init__(
        self,
        detector: RPeakDetector | None = None,
        timing: DetectorTiming | None = None,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        self._buffer = IndexedSampleBuffer()
        self._detector = StreamingXQRS(detector=detector, timing=timing)
        self._assembler = SequenceAssembler(sequence_length=sequence_length)
        self._state = StreamState()
        # tuple[int,...] means the uple can contain any number of elements
        # and they must be ints
        self._last_confirmed_peaks: tuple[int, ...] = ()

        logger.debug(
            "Initialised streaming engine with sequence length %s",
            sequence_length,
        )

    @property
    def state(self) -> StreamState:
        """Current stream state, scoped to the record being processed."""

        return self._state

    @property
    def last_confirmed_peaks(self) -> tuple[int, ...]:
        """
        Peaks confirmed by the most recent process_chunk or flush call.

        Only the latest call is retained, so nothing grows without bound.
        Callers that want the whole record's detection timeline accumulate
        it themselves.
        """

        return self._last_confirmed_peaks

    def start_record(
        self,
        record_name: str | None = None,
        start_index: int = 0,
    ) -> None:
        """
        Begin a new record, discarding any previous stream state.

        Record boundaries are explicit so no counters, sampling rate,
        expected sample position, ECG history, detector state, pending
        peak, RR history or completed beat can leak between records.

        start_index is the Absolute index the first chunk of this
        record must start at. Defaults to zero. We supply it when a
        streaming deliberately begins part-way through a record,
        so callers never have to modify continuity state by hand.
        """

        # Where we want to start in the record
        start_index = validate_start_index(start_index)

        # Create the state object for this record
        self._state = StreamState(
            record_name=record_name,
            next_expected_index=start_index,
        )
        # Essentially resets streaming engine
        self._buffer.reset(start_index)
        self._detector.reset()
        self._assembler.reset(start_index)
        self._last_confirmed_peaks = ()

        logger.debug(
            "Started streaming record %s at sample %s",
            record_name,
            start_index,
        )

    def reset(self) -> None:
        """Discard all stream state without naming a new record."""

        self.start_record(record_name=None, start_index=0)

    def process_chunk(self, chunk: SampleChunk) -> list[BeatSequence]:
        """
        Accept the next chunk of the current stream.

        The chunk must start exactly where the previous one ended, must
        carry the same sampling rate as the rest of the record, and must
        be sampled at the rate the model was trained on. Gaps, overlaps,
        duplicates, out-of-order chunks and unusable sampling rates are
        rejected before anything downstream is touched, so a rejected
        chunk cannot leave the buffer, detector or assembler partly
        advanced.

        Returns the sequences completed by this chunk, which may be none,
        one, or several.
        """

        logger.debug(
            "Received chunk for record %s: start=%s, stop=%s, samples=%s, "
            "sampling_rate=%s",
            self._state.record_name,
            chunk.start_index,
            chunk.stop_index,
            chunk.num_samples,
            chunk.sampling_rate,
        )

        # Validates if this chunks sampling rate matches
        # the StreamState sampling rate for the record it is in
        self._validate_sampling_rate(chunk)

        # Checks that the stream can produce valid model inputs at all.
        self._validate_model_sampling_rate(chunk)

        # Checks that this next chunk is valid to run. I.e.,
        # it is not a duplicate, starts in the right place,
        # does not overlap, etc.
        self._validate_continuity(chunk)

        # If we have not seen a chunk yet, make its start
        # index equal to the start index of the first chunk
        if self._state.first_sample_index is None:
            self._state.first_sample_index = chunk.start_index

        # This will be the sampling rate used for this entire record.
        if self._state.sampling_rate is None:
            self._state.sampling_rate = chunk.sampling_rate

        # Update this records state.
        self._state.next_expected_index = chunk.stop_index
        self._state.total_samples_accepted += chunk.num_samples
        self._state.num_chunks_accepted += 1

        # Appends this chunks samples to the buffer.
        self._buffer.append(chunk.samples, chunk.start_index)

        logger.debug(
            "Accepted chunk %s for record %s: total_chunks=%s, "
            "total_samples=%s, next_expected_index=%s",
            self._state.num_chunks_accepted,
            self._state.record_name,
            self._state.num_chunks_accepted,
            self._state.total_samples_accepted,
            self._state.next_expected_index,
        )

        #
        sequences = self._advance(chunk.sampling_rate, end_of_record=False)
        self._prune(chunk.sampling_rate)

        logger.debug(
            "Completed chunk for record %s: confirmed_peaks=%s, "
            "sequences_emitted=%s",
            self._state.record_name,
            len(self._last_confirmed_peaks),
            len(sequences),
        )

        return sequences

    def flush(self) -> list[BeatSequence]:
        """
        Release the outputs that can still be completed at end of record.

        Peaks whose beat window runs past the final sample, and beats that
        never reach a full sequence, are dropped rather than padded.
        """

        sampling_rate = self._state.sampling_rate

        if sampling_rate is None:
            logger.debug(
                "Flush requested with no accepted samples for record %s",
                self._state.record_name,
            )
            return []

        logger.debug(
            "Flushing record %s after %s chunks and %s samples",
            self._state.record_name,
            self._state.num_chunks_accepted,
            self._state.total_samples_accepted,
        )

        sequences = self._advance(sampling_rate, end_of_record=True)

        logger.debug(
            "Finished flushing record %s: confirmed_peaks=%s, "
            "sequences_emitted=%s",
            self._state.record_name,
            len(self._last_confirmed_peaks),
            len(sequences),
        )

        return sequences

    def _advance(
        self,
        sampling_rate: float,
        end_of_record: bool,
    ) -> list[BeatSequence]:
        # detect peaks in the current buffer
        peaks = self._detector.confirmed_peaks(
            buffer=self._buffer,
            sampling_rate=sampling_rate,
            end_of_record=end_of_record,
        )
        self._last_confirmed_peaks = tuple(peaks)

        logger.debug(
            "Detector advance for record %s: end_of_record=%s, peaks=%s",
            self._state.record_name,
            end_of_record,
            self._last_confirmed_peaks,
        )

        # Add the peaks into pending awaiting sequencing
        self._assembler.add_peaks(peaks)

        # From these pending peaks, we can create beats, and when we have enough
        # beats we can create sequences.
        sequences = self._assembler.drain(
            self._buffer,
            end_of_record=end_of_record,
        )

        logger.debug(
            "Assembler advance for record %s: sequences=%s, target_peaks=%s",
            self._state.record_name,
            len(sequences),
            tuple(sequence.target_peak_index for sequence in sequences),
        )

        return sequences

    def _prune(self, sampling_rate: float) -> None:
        """Drop history neither the detector nor a pending beat can need."""

        previous_start = self._buffer.start_index

        # This is the oldest sample we need for the analysis window
        keep_from = self._detector.history_start(
            stop_index=self._buffer.stop_index,
            sampling_rate=sampling_rate,
        )
        # This is the sample needed to retain a window around the
        # oldest peak
        pending_start = self._assembler.history_start

        if pending_start is not None:
            # We set keep from to be the smallest out of the two
            keep_from = min(keep_from, pending_start)

        # We remove all samples in the buffer from index keep_from and
        # below to the start of the ECG range. Giving us a new range starting
        # from keep_from up to stop_index.
        self._buffer.prune_before(keep_from)

        logger.debug(
            "Pruned buffer for record %s: previous_start=%s, "
            "new_start=%s, stop=%s, pending_start=%s",
            self._state.record_name,
            previous_start,
            self._buffer.start_index,
            self._buffer.stop_index,
            pending_start,
        )

    def _validate_model_sampling_rate(self, chunk: SampleChunk) -> None:
        """
        Reject input the trained beat and RR contract cannot describe.

        The 90/150-sample beat window and the RR features are defined at
        the preprocessing sampling rate, so a stream at any other rate
        would produce sequences the model was never trained to read. The
        engine refuses these inputs rather than resampling it.
        """

        if chunk.sampling_rate != MODEL_SAMPLING_RATE:
            logger.warning(
                "Rejected chunk for record %s because its sampling rate is %s Hz; "
                "the model requires %s Hz",
                self._state.record_name,
                chunk.sampling_rate,
                MODEL_SAMPLING_RATE,
            )
            raise ValueError(
                f"Streaming requires {MODEL_SAMPLING_RATE} Hz input because "
                "the beat window and RR features are defined at that rate, "
                f"but received {chunk.sampling_rate} Hz. Resample the source "
                "before streaming it."
            )

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
            logger.warning(
                "Rejected chunk for record %s because the sampling rate changed "
                "from %s Hz to %s Hz",
                self._state.record_name,
                stream_sampling_rate,
                chunk.sampling_rate,
            )
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
            logger.warning(
                "Rejected chunk for record %s because of a %s-sample gap: "
                "expected_start=%s, received_start=%s",
                self._state.record_name,
                missing_samples,
                expected_index,
                chunk.start_index,
            )
            raise StreamContinuityError(
                f"Gap of {missing_samples} samples: expected a chunk "
                f"starting at {expected_index} but received one starting "
                f"at {chunk.start_index}."
            )

        # If the stop index of the next chunk is equal to the expected index,
        # then this is the chunk that we just processed, so it has been
        # duplicated and thus should be rejected.
        if chunk.stop_index == expected_index:
            logger.warning(
                "Rejected duplicate chunk for record %s: start=%s, stop=%s, "
                "stream_position=%s",
                self._state.record_name,
                chunk.start_index,
                chunk.stop_index,
                expected_index,
            )
            raise StreamContinuityError(
                f"Duplicate chunk covering samples {chunk.start_index} to "
                f"{chunk.last_index}: the stream already reached "
                f"{expected_index}."
            )

        # A chunk starting before the expected position is either the
        # previous chunk resent, or an overlapping/out-of-order chunk.
        logger.warning(
            "Rejected overlapping or out-of-order chunk for record %s: "
            "expected_start=%s, received_start=%s, received_stop=%s",
            self._state.record_name,
            expected_index,
            chunk.start_index,
            chunk.stop_index,
        )
        raise StreamContinuityError(
            "Overlapping or out-of-order chunk: expected a chunk starting "
            f"at {expected_index} but received one starting at "
            f"{chunk.start_index}."
        )