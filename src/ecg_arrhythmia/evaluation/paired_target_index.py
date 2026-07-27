from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# A stable target identity is the expert ground-truth heartbeat: the
# record it belongs to and its expert annotation sample position. Both the
# expert-centred and XQRS-centred pipelines can produce this identity.
TargetKey = tuple[str, int]


@dataclass(frozen=True)
class PairedTargetIndex:
    """Row alignment between expert-centred and XQRS-centred targets."""

    paired_keys: list[TargetKey]
    expert_rows: NDArray[np.int64]
    xqrs_rows: NDArray[np.int64]

    expert_only_keys: list[TargetKey]
    xqrs_only_keys: list[TargetKey]

    num_expert_targets: int
    num_xqrs_targets: int

    @property
    def num_paired(self) -> int:
        return len(self.paired_keys)


@dataclass(frozen=True)
class PairedViews:
    """Aligned expert-centred and XQRS-centred inputs for shared targets."""

    expert_X: NDArray[np.float64]
    expert_rr: NDArray[np.float64]
    expert_y: NDArray[np.str_]

    xqrs_X: NDArray[np.float64]
    xqrs_rr: NDArray[np.float64]
    xqrs_y: NDArray[np.str_]

    # Shared per-target audit (order matches both views).
    records: NDArray[np.str_]
    annotation_samples: NDArray[np.int64]
    offset_samples: NDArray[np.int64]
    offset_ms: NDArray[np.float64]
    has_unmatched_context: NDArray[np.bool_]


def _build_identity_index(
    records: NDArray[np.str_],
    annotation_samples: NDArray[np.int64],
    side_name: str,
) -> dict[TargetKey, int]:
    """
    Map each target identity to its row, rejecting duplicate identities.

    Pairing must never rely on array position or label alone, so a stable
    identity is required and must be unique within each side.
    """

    if records.shape[0] != annotation_samples.shape[0]:
        raise ValueError(
            f"{side_name}: records and annotation samples differ in length."
        )

    index: dict[TargetKey, int] = {}
    for row in range(records.shape[0]):
        key: TargetKey = (str(records[row]), int(annotation_samples[row]))
        if key in index:
            raise ValueError(
                f"Duplicate {side_name} target identity {key}. Target "
                "identities must be unique."
            )
        index[key] = row

    return index


def build_paired_target_index(
    expert_records: NDArray[np.str_],
    expert_annotation_samples: NDArray[np.int64],
    xqrs_records: NDArray[np.str_],
    xqrs_annotation_samples: NDArray[np.int64],
) -> PairedTargetIndex:
    """
    Build the paired target index from the intersection of expert-centred
    and XQRS-centred target identities.

    Ordering is deterministic (sorted by identity), duplicates are
    rejected, and only shared targets are retained.
    """

    expert_index = _build_identity_index(
        expert_records, expert_annotation_samples, "expert"
    )
    xqrs_index = _build_identity_index(xqrs_records, xqrs_annotation_samples, "xqrs")

    expert_keys = set(expert_index)
    xqrs_keys = set(xqrs_index)

    paired_keys = sorted(expert_keys & xqrs_keys)

    expert_rows = np.asarray([expert_index[key] for key in paired_keys], dtype=np.int64)
    xqrs_rows = np.asarray([xqrs_index[key] for key in paired_keys], dtype=np.int64)

    return PairedTargetIndex(
        paired_keys=paired_keys,
        expert_rows=expert_rows,
        xqrs_rows=xqrs_rows,
        expert_only_keys=sorted(expert_keys - xqrs_keys),
        xqrs_only_keys=sorted(xqrs_keys - expert_keys),
        num_expert_targets=len(expert_index),
        num_xqrs_targets=len(xqrs_index),
    )


def create_paired_dataset_views(
    paired_index: PairedTargetIndex,
    expert_X: NDArray[np.float64],
    expert_rr: NDArray[np.float64],
    expert_y: NDArray[np.str_],
    xqrs_X: NDArray[np.float64],
    xqrs_rr: NDArray[np.float64],
    xqrs_y: NDArray[np.str_],
    xqrs_records: NDArray[np.str_],
    xqrs_annotation_samples: NDArray[np.int64],
    xqrs_offset_samples: NDArray[np.int64],
    xqrs_offset_ms: NDArray[np.float64],
    xqrs_has_unmatched_context: NDArray[np.bool_],
) -> PairedViews:
    """
    Select and align the expert-centred and XQRS-centred inputs for the
    shared targets.

    Both views end up with identical target identities, labels, ordering,
    and count. The label agreement is asserted because a mismatch would
    indicate a broken identity mapping.
    """

    expert_rows = paired_index.expert_rows
    xqrs_rows = paired_index.xqrs_rows

    expert_labels = expert_y[expert_rows].astype(str)
    xqrs_labels = xqrs_y[xqrs_rows].astype(str)

    if not np.array_equal(expert_labels, xqrs_labels):
        mismatches = int(np.sum(expert_labels != xqrs_labels))
        raise ValueError(
            f"Paired expert and XQRS labels disagree for {mismatches} "
            "targets. The target identity mapping is inconsistent."
        )

    return PairedViews(
        expert_X=expert_X[expert_rows],
        expert_rr=expert_rr[expert_rows],
        expert_y=expert_labels,
        xqrs_X=xqrs_X[xqrs_rows],
        xqrs_rr=xqrs_rr[xqrs_rows],
        xqrs_y=xqrs_labels,
        records=xqrs_records[xqrs_rows].astype(str),
        annotation_samples=xqrs_annotation_samples[xqrs_rows].astype(np.int64),
        offset_samples=xqrs_offset_samples[xqrs_rows].astype(np.int64),
        offset_ms=xqrs_offset_ms[xqrs_rows].astype(np.float64),
        has_unmatched_context=xqrs_has_unmatched_context[xqrs_rows].astype(bool),
    )


def build_pairing_summary(
    paired_index: PairedTargetIndex,
    paired_labels: NDArray[np.str_],
) -> dict[str, object]:
    """Build the compact JSON-serialisable pairing summary."""

    paired_support = {
        str(label): int(count)
        for label, count in sorted(Counter(paired_labels.tolist()).items())
    }

    return {
        "total_expert_targets": paired_index.num_expert_targets,
        "total_xqrs_targets": paired_index.num_xqrs_targets,
        "paired_targets": paired_index.num_paired,
        "expert_only_excluded": len(paired_index.expert_only_keys),
        "xqrs_only_excluded": len(paired_index.xqrs_only_keys),
        "paired_class_support": paired_support,
    }
