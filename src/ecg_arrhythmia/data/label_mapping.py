import numpy as np

# Following the Association for the Advancement of Medical Instrumentation (AAMI)
# grouping, raw MIT-BIH beat annotations can be condensed into 5 classes. This helps
# with the huge class imbalance observed in the dataset summary.
#
# N: normal and bundle branch block beats
# S: supraventricular ectopic beats
# V: ventricular ectopic beats
# F: fusion beats
# Q: unknown, paced, or unclassifiable beats

# fmt: off
AAMI_LABEL_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    "V": "V",
    "E": "V",

    "F": "F",

    "Q": "Q",
    "?": "Q",
    "/": "Q",
    "f": "Q"
}
# fmt: on


# Shared model class-index mappings.
LABEL_TO_INDEX = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
}

INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(LABEL_TO_INDEX)
CLASS_LABELS = [INDEX_TO_LABEL[index] for index in range(NUM_CLASSES)]
CLASS_INDICES = list(range(NUM_CLASSES))


def map_labels_to_aami(labels: np.ndarray) -> np.ndarray:
    """
    Map raw MIT-BIH beat annotation symbols to AAMI classes.
    """

    # This gets the unique labels in each and sees if there are any leftover.
    # This should be the empty set if there are no labels in labels that are not
    # in the AAMI MAP
    unknown_labels = set(labels.tolist()) - set(AAMI_LABEL_MAP)

    if unknown_labels:
        raise ValueError(f"Unmapped labels found: {unknown_labels}")

    # Iterate through labels and map each label to its AAMI map
    return np.array(
        [AAMI_LABEL_MAP[label] for label in labels],
        dtype=str,
    )
