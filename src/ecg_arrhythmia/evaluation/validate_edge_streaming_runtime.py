import argparse
import json
import logging
import platform
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ecg_arrhythmia.data.label_mapping import (
    CLASS_LABELS,
    LABEL_TO_INDEX,
    NUM_CLASSES,
)
from ecg_arrhythmia.streaming.onnx_sequence_classifier import (
    ONNXSequenceClassifier,
    PredictionEvent,
)
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE, ReplaySource
from ecg_arrhythmia.streaming.sample_chunk import SampleChunk
from ecg_arrhythmia.streaming.streaming_engine import StreamingEngine
from ecg_arrhythmia.streaming.streaming_predictor import StreamingPredictor
from ecg_arrhythmia.telemetry.edge_sensors import (
    parse_temperature,
    parse_throttled,
    read_meminfo,
    run_vcgencmd,
)

logger = logging.getLogger(__name__)

DEFAULT_RECORD_NAME = "114"
DEFAULT_FP32_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer.onnx")
DEFAULT_INT8_MODEL_PATH = Path("artifacts/models/ecg_sequence_transformer_int8.onnx")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/results/deployment_evaluation/edge_runtime_validation"
)

MEMINFO_PATH = Path("/proc/meminfo")
KIB_PER_MIB = 1024

# How many integrity failures to describe verbatim before only counting.
MAX_RECORDED_FAILURES = 5


# ---------------------------------------------------------------------
#                          Event Accumulation
# ---------------------------------------------------------------------


class EventAccumulator:
    """
    Lightweight per-run statistics over a stream of PredictionEvents.

    Only the target peak index and predicted label of each event are
    retained (a few bytes per beat). Logits, peak tuples and sequences
    are checked and then discarded, which is what keeps the harness
    honest about edge memory behaviour.
    """

    def __init__(self) -> None:
        self.class_counts: dict[str, int] = {label: 0 for label in CLASS_LABELS}
        self.target_peaks: list[int] = []
        self.predicted_labels: list[str] = []
        self.flush_event_count = 0
        self.flush_called = False
        self.integrity_failure_count = 0
        self.integrity_failures: list[str] = []

    @property
    def num_events(self) -> int:
        return len(self.target_peaks)

    @property
    def first_target_peak(self) -> int | None:
        return self.target_peaks[0] if self.target_peaks else None

    @property
    def last_target_peak(self) -> int | None:
        return self.target_peaks[-1] if self.target_peaks else None

    @property
    def integrity_passed(self) -> bool:
        return self.integrity_failure_count == 0

    def add_events(
        self,
        events: Iterable[PredictionEvent],
        from_flush: bool = False,
    ) -> None:
        """Record a batch of events, checking invariants on each."""

        if from_flush:
            self.flush_called = True

        for event in events:
            # Records the failures, if any, of the event
            self._check_integrity(event)

            # Add the target peaks and labels to the accumulator
            self.target_peaks.append(int(event.target_peak_index))
            self.predicted_labels.append(event.predicted_label)

            # increment this predicted class labels count
            if event.predicted_label in self.class_counts:
                self.class_counts[event.predicted_label] += 1

            # If this event is from a flush, update the flush count
            if from_flush:
                self.flush_event_count += 1

    def _check_integrity(self, event: PredictionEvent) -> None:
        failures = []

        if event.predicted_label not in LABEL_TO_INDEX:
            failures.append(f"unknown label {event.predicted_label!r}")

        if not 0 <= event.predicted_class_index < NUM_CLASSES:
            failures.append(f"class index {event.predicted_class_index} out of range")

        # Predicted label should match its corresponding index
        elif LABEL_TO_INDEX.get(event.predicted_label) != event.predicted_class_index:
            failures.append(
                f"label {event.predicted_label!r} does not match class "
                f"index {event.predicted_class_index}"
            )

        logits = np.asarray(event.logits)

        # There should be on logit value per class
        if logits.shape != (NUM_CLASSES,):
            failures.append(f"logits shape {logits.shape} != ({NUM_CLASSES},)")
        # They should all be finite.
        elif not np.all(np.isfinite(logits)):
            failures.append("non-finite logits")

        # If the running target peaks accumulated by this class is non-empty,
        # and the event we are trying to add has a peak that comes before the
        # last seen peak, then it is out of order, and hence an error.
        if self.target_peaks and event.target_peak_index <= self.target_peaks[-1]:
            failures.append(
                f"target peak {event.target_peak_index} not strictly after "
                f"{self.target_peaks[-1]}"
            )

        # Sum the number of failures, if any
        for failure in failures:
            self.integrity_failure_count += 1

            # If the number of failures is less than the max recorded failures
            if len(self.integrity_failures) < MAX_RECORDED_FAILURES:
                # Then we the full failure to integrity failures, with the peak
                # it was associated with.
                self.integrity_failures.append(
                    f"target peak {event.target_peak_index}: {failure}"
                )


def stream_and_accumulate(
    predictor: StreamingPredictor,
    Generator: Iterable[SampleChunk],
) -> tuple[EventAccumulator, int]:
    """
    Drive the production predictor chunk by chunk, then flush exactly once.

    Events are folded into the accumulator as they are emitted and never
    collected into a full-record list, mirroring real deployment.
    """

    # Create the accumulator for this record
    accumulator = EventAccumulator()
    num_chunks = 0

    # Adds information like class counts, predicted labels,
    # target peaks, failures, etc.
    for chunk in Generator:
        accumulator.add_events(predictor.process_chunk(chunk))
        num_chunks += 1

    # Adds the events from the flush
    accumulator.add_events(predictor.flush(), from_flush=True)

    return accumulator, num_chunks


def run_model_validation(
    precision: str,
    model_path: Path,
    source: ReplaySource,
) -> tuple[dict, EventAccumulator, tuple[str, ...]]:
    """
    Run one full record through the production path with one model.

    A fresh StreamingEngine and classifier are constructed per run so the
    FP32 and INT8 passes are fully independent; the classifier is
    released when this function returns, so both sessions are never
    resident together.
    """

    logger.info("Validating %s model %s", precision.upper(), model_path)

    # Create the classifier
    classifier = ONNXSequenceClassifier(model_path)
    # Create the Streaming predicitor
    predictor = StreamingPredictor(
        engine=StreamingEngine(),
        classifier=classifier,
    )
    # Resets the streaming pipeline and intialises it ready
    # for this new record.
    predictor.start_record(record_name=source.record_name)

    # Returns the accumulator which now has this records
    # information and the number of chunks,
    accumulator, num_chunks = stream_and_accumulate(
        predictor,
        source.iter_chunks(),
    )
    # Returns the StreamState object that tracked this run
    state = predictor.engine.state

    summary = {
        "precision": precision,
        "model_path": str(model_path),
        "chunks_processed": num_chunks,
        "chunks_accepted": state.num_chunks_accepted,
        "samples_accepted": state.total_samples_accepted,
        "prediction_events": accumulator.num_events,
        "class_counts": dict(accumulator.class_counts),
        "first_target_peak": accumulator.first_target_peak,
        "last_target_peak": accumulator.last_target_peak,
        "flush_called": accumulator.flush_called,
        "flush_prediction_events": accumulator.flush_event_count,
        "integrity": {
            "passed": accumulator.integrity_passed,
            "failure_count": accumulator.integrity_failure_count,
            "failures": list(accumulator.integrity_failures),
        },
    }

    return summary, accumulator, classifier.providers


# ---------------------------------------------------------------------
#                    FP32 vs INT8 Lightweight Comparison
# ---------------------------------------------------------------------


def compare_runs(
    fp32: EventAccumulator,
    int8: EventAccumulator,
) -> dict:
    """
    Confirm both runs traversed the same streaming events.

    This is a sanity check that the two passes saw identical beats, not a
    rerun of Section 4.3's numerical agreement study. Class agreement is
    reported but carries no threshold: quantisation legitimately changes
    a small number of decisions.
    """

    # Tells us is both models received/traversed the same sequence stream
    # and produced predictions for the same target beats.
    targets_identical = fp32.target_peaks == int8.target_peaks

    comparison = {
        "fp32_prediction_events": fp32.num_events,
        "int8_prediction_events": int8.num_events,
        "target_peaks_identical": targets_identical,
        "class_agreements": None,
        "class_disagreements": None,
        "class_agreement_percentage": None,
    }

    # Comparison only makes sense it that targets are identical,
    # and they produced atleast one event.
    if targets_identical and fp32.num_events > 0:
        # Firs combines each label from each model into a list
        # of tuples, then returns True if they match, False otherwise,
        # and lastly sums over the values. This tells us on how many
        # events did they agree
        agreements = sum(
            fp32_label == int8_label
            for fp32_label, int8_label in zip(
                fp32.predicted_labels,
                int8.predicted_labels,
                strict=True,
            )
        )
        # Update the comparison dictionary to include agreements, disagreements,
        # and agreement percentage.
        comparison["class_agreements"] = agreements
        comparison["class_disagreements"] = fp32.num_events - agreements
        comparison["class_agreement_percentage"] = agreements / fp32.num_events * 100.0

    return comparison


# ---------------------------------------------------------------------
#                     Hardware Health (Pi, graceful)
# ---------------------------------------------------------------------
#


def health_snapshot(
    meminfo_path: Path = MEMINFO_PATH,
    vcgencmd: Callable[[str], str | None] = run_vcgencmd,
) -> dict:
    """
    One point-in-time hardware health reading.

    Every field degrades independently to None, so a missing Pi command
    never breaks the run and a WSL test environment reports what it can.
    """

    # This returns our PIs total ram and availble rma in mib
    memory = read_meminfo(meminfo_path)

    return {
        "total_ram_mib": memory["total_ram_mib"],
        "available_ram_mib": memory["available_ram_mib"],
        # vcgencmd("measure_temp") Is a Raspberry Pi command-line
        # utility that lets you ask the Pi's firmware for hardware
        # information. So we can ask for the temperature and throttle
        # information.
        # Returns the raw temperature value
        "temperature_c": parse_temperature(vcgencmd("measure_temp")),
        # Returns the throttled flag
        "throttled": parse_throttled(vcgencmd("get_throttled")),
    }


# ---------------------------------------------------------------------
#                        Environment And Status
# ---------------------------------------------------------------------


def runtime_environment(providers: tuple[str, ...]) -> dict:
    """
    Machine and runtime details from the standard library and onnxruntime.

    Deliberately local rather than imported from the Section 4 benchmark
    module, whose import chain pulls in desktop-only dependencies the
    edge runtime does not ship.
    """

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "execution_providers": list(providers),
        "provider": providers[0] if providers else None,
    }


def overall_status(
    fp32_summary: dict,
    int8_summary: dict,
    comparison: dict,
) -> dict:
    """
    PASS/FAIL for the runtime validation, with explicit reasons.

    Health observations are deliberately excluded: correctness and
    hardware condition are reported side by side but never conflated.
    Class disagreement is not a failure; mismatched target peaks are.
    """

    reasons = []

    for summary in (fp32_summary, int8_summary):
        precision = summary["precision"].upper()

        # If this model made no predictions, this is an error
        if summary["prediction_events"] == 0:
            reasons.append(f"{precision} emitted no PredictionEvents")

        # If we did not capture the remaining events due to not calling
        # flush, that is an error
        if not summary["flush_called"]:
            reasons.append(f"{precision} flush was never called")

        # And there were any failures caught in the accumulator, that is
        # an error
        if not summary["integrity"]["passed"]:
            reasons.append(
                f"{precision} failed {summary['integrity']['failure_count']} "
                "event integrity checks"
            )

    if not comparison["target_peaks_identical"]:
        reasons.append("FP32 and INT8 traversed different target peaks")

    # If no errors were appended to reasons, then PASSED, else FAILED
    return {
        "status": "PASSED" if not reasons else "FAILED",
        "reasons": reasons,
    }


# ---------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the production streaming inference path on target "
            "hardware with FP32 and INT8 ONNX models."
        )
    )
    parser.add_argument("--record-name", type=str, default=DEFAULT_RECORD_NAME)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--fp32-model-path",
        type=Path,
        default=DEFAULT_FP32_MODEL_PATH,
    )
    parser.add_argument(
        "--int8-model-path",
        type=Path,
        default=DEFAULT_INT8_MODEL_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Extract command line arguments
    args = parse_args()

    # One record load shared by both runs; iter_chunks() yields a fresh
    # pass over the same signal each time it is called.
    source = ReplaySource.from_record(
        record_name=args.record_name,
        chunk_size=args.chunk_size,
    )

    # Returns the total ram, available ram, temperature, and throttled flag
    # before model validation.
    health_before = health_snapshot()

    # Returns the summary of a record running from this the fp32 ONNX model
    # on the device, along with its accumulator and provider.
    fp32_summary, fp32_accumulator, providers = run_model_validation(
        "fp32",
        args.fp32_model_path,
        source,
    )
    # Returns the summary of a record running from this the INT8 ONNX model
    # on the device, along with its accumulator and provider.
    int8_summary, int8_accumulator, _ = run_model_validation(
        "int8",
        args.int8_model_path,
        source,
    )

    # Returns the total ram, available ram, temperature, and throttled flag
    # after model validation
    health_after = health_snapshot()

    # Compares the events accumulated for fp32 and int8, and returns
    # how many events that ran, if their target peak indices were identical,
    # and their agreements, disagreements, and agreement percentange.
    comparison = compare_runs(fp32_accumulator, int8_accumulator)

    # Returns if runtime PASSES or FAILED
    validation = overall_status(fp32_summary, int8_summary, comparison)

    result = {
        "environment": runtime_environment(providers),
        "record": {
            "record_name": args.record_name,
            "lead_name": source.lead_name,
            "sampling_rate": source.sampling_rate,
            "num_samples": source.num_samples,
            "num_chunks": source.num_chunks,
        },
        "stream_configuration": {
            "chunk_size": args.chunk_size,
            "replay_mode": "accelerated",
        },
        "fp32_run": fp32_summary,
        "int8_run": int8_summary,
        "comparison": comparison,
        "hardware_health": {"before": health_before, "after": health_after},
        "runtime_validation": validation,
    }

    output_path = (
        args.output_dir / f"record_{args.record_name}_edge_runtime_validation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    logger.info("Wrote runtime validation to %s", output_path)

    if validation["status"] != "PASSED":
        sys.exit(1)


if __name__ == "__main__":
    main()
