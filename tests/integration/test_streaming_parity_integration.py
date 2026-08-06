import pytest

from ecg_arrhythmia.evaluation.evaluate_streaming_parity import (
    aggregate_records,
    evaluate_record,
)
from ecg_arrhythmia.streaming.replay_source import DEFAULT_CHUNK_SIZE

# 114 reaches exact parity, which makes it the strictest single-record
# check that streaming reproduces the offline XQRS-centred dataset.
EXACT_PARITY_RECORD = "114"


@pytest.mark.integration
def test_record_114_reaches_exact_streaming_parity():
    result = evaluate_record(
        record_name=EXACT_PARITY_RECORD,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )

    assert result["continuity_validated"] is True
    assert result["samples_accepted"] == result["total_input_samples"]

    # The causal detector must find the same peaks in the same places.
    assert result["missing_peaks"] == []
    assert result["extra_peaks"] == []
    assert result["largest_absolute_peak_offset"] == 0

    # Every offline target is reproduced with identical ECG and RR values.
    assert result["missing_targets"] == []
    assert result["extra_targets"] == []
    assert result["ecg_window_mismatches"] == []
    assert result["rr_feature_mismatches"] == []
    assert result["exactly_matched_sequences"] == result["offline_targets_expected"]
    assert result["exact_parity"] is True

    aggregate = aggregate_records([result])
    assert aggregate["all_records_exact_parity"] is True
    assert aggregate["all_records_differences_explained"] is True
