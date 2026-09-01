# Progress Log

## Milestone 1 — MIT-BIH data loading and beat extraction

Implemented:

- Added support for loading ECG records from the MIT-BIH Arrhythmia Database using WFDB.
- Loaded both ECG signal channels and expert beat annotations for each record.
- Added signal-channel selection, preferring `MLII` when available and falling back to the first available channel otherwise.
- Implemented beat extraction around annotation locations.
- Each beat is represented as a fixed 240-sample ECG window:
  - 90 samples before the annotation
  - 150 samples after the annotation
- Filtered out annotations that do not represent heartbeat classes.

Key lesson:

- ECG data needs careful preprocessing before it can be used by a neural network.
- Beat-level classification requires each annotation to be converted into a fixed-size input window.
- Keeping the window size fixed makes the later PyTorch dataset and CNN models much simpler.
- Some beats near the start or end of a record cannot produce a complete window and must be skipped.

## Milestone 2 — AAMI label mapping and dataset building

Implemented:

- Added AAMI-style label mapping to group raw MIT-BIH beat annotations into higher-level arrhythmia classes.
- Mapped raw annotations into:
  - `N`
  - `S`
  - `V`
  - `F`
  - `Q`
- Excluded the `Q` class during dataset build due to low support and ambiguity.
- Built the processed dataset using 4 classes:
  - `N`
  - `S`
  - `V`
  - `F`
- Saved processed arrays to disk:
  - `X.npy`
  - `y.npy`
  - `patient_ids.npy`
  - `record_segments.json`
- Stored record-level metadata including:
  - record ID
  - patient ID
  - selected ECG lead
  - start index
  - end index
  - number of extracted beats

Key lesson:

- Raw ECG annotation labels are too fragmented and imbalanced to use directly.
- AAMI grouping makes the classification problem more manageable while still preserving clinically meaningful arrhythmia categories.
- Saving `patient_ids.npy` is important because the dataset must later be split by patient, not by individual beat.
- Storing record metadata makes the processed dataset easier to inspect and debug.

## Milestone 3 — Patient-level train/validation/test splitting

Implemented:

- Created patient-level train, validation, and test splits.
- Ensured that no patient appears in more than one split.
- Added split validation to prevent patient leakage.
- Used Monte Carlo sampling to search for a split that approximately preserves:
  - target train/validation/test ratios
  - class distribution across splits
- Saved each split separately with its own:
  - `X.npy`
  - `y.npy`
  - `patient_ids.npy`
  - metadata

Key lesson:

- Patient-level splitting is essential for ECG classification.
- Random beat-level splitting would leak patient-specific ECG morphology into multiple splits and produce skewed, overly optimistic results.
- The split cannot be perfectly balanced because patients contain different numbers and types of beats.
- Monte Carlo split search gives a practical way to find a reasonable patient-safe split.

## Milestone 4 — Testing and continuous integration

Implemented:

- Added unit tests for beat extraction.
- Added unit tests for AAMI label mapping.
- Added unit tests for ECG record loading and signal-channel selection.
- Added unit tests for dataset building.
- Added unit tests for patient-level splitting.
- Added integration tests using real MIT-BIH records.
- Added GitHub Actions CI.
- Configured CI to run automated checks on pushes and pull requests.

Key lesson:

- Tests are especially useful in data pipelines because small preprocessing bugs can silently corrupt the whole dataset.
- Unit tests make individual behaviours easier to verify without depending on the full MIT-BIH database.
- Integration tests confirm that the real WFDB/MIT-BIH pipeline works end-to-end.
- CI makes the repository safer to refactor because tests are run automatically.

## Milestone 5 — PyTorch dataset and CNN Baseline V1

Implemented:

- Created a PyTorch `ECGDataset`.
- Loaded processed split data from disk.
- Converted string labels into integer class indices:
  - `N -> 0`
  - `S -> 1`
  - `V -> 2`
  - `F -> 3`
- Returned ECG windows in the shape expected by `Conv1d`:

```text
(channels, sequence_length) = (1, 240)
```

- Implemented `CNNBaselineV1`, a simple 1D CNN with:
  - convolution layers
  - ReLU activations
  - max pooling
  - adaptive average pooling
  - final linear classifier

Key lesson:

- `Conv1d` expects the channel dimension before the sequence dimension, so each ECG window must be reshaped from `(240,)` to `(1, 240)`.
- A simple CNN is a useful first baseline because it proves the full pipeline works from processed ECG windows to class predictions.
- The first model does not need to be perfect; it mainly establishes a measurable reference point.

## Milestone 6 — Weighted training and evaluation pipeline

Implemented:

- Added a reusable CNN training script.
- Added weighted `CrossEntropyLoss` to handle class imbalance.
- Computed class weights from the training set.
- Added validation after each epoch.
- Saved the best model checkpoint based on validation macro F1.
- Added evaluation metrics:
  - loss
  - accuracy
  - macro F1
  - per-class precision
  - per-class recall
  - per-class F1
  - confusion matrix
- Added structured confusion matrix output for easier JSON inspection.

Key lesson:

- Accuracy is misleading for this dataset because `N` dominates the class distribution.
- Macro F1 is a better main metric because it gives each class equal importance.
- Per-class metrics and confusion matrices are necessary because the model can appear reasonable overall while completely failing minority arrhythmia classes.
- Weighted loss helps force the model to pay attention to rare classes, but it can also cause overprediction of minority classes.

## Milestone 7 — CNN Baseline V1 test evaluation

Implemented:

- Trained and evaluated `CNNBaselineV1`.
- Saved the model checkpoint to:

```text
artifacts/models/cnn_baseline_v1.pt
```

- Saved test metrics to:

```text
artifacts/results/cnn_baseline_v1_test_metrics.json
```

Test results:

| Metric | Value |
|---|---:|
| Test loss | 1.2953 |
| Test accuracy | 0.7177 |
| Test macro F1 | 0.2807 |

Per-class test F1:

| Class | F1 |
|---|---:|
| N | 0.8372 |
| S | 0.0287 |
| V | 0.2450 |
| F | 0.0120 |

Key lesson:

- CNN V1 learned the majority `N` class well but struggled heavily with minority classes.
- `V` was detected better than `S` and `F`, but still had poor precision.
- `S` and `F` were almost unusable in the first baseline.

## Milestone 8 — CNN Baseline V2 and shared model training

Implemented:

- Added `CNNBaselineV2`, a stronger CNN architecture.
- Increased the number of convolution channels.
- Added `BatchNorm1d` after convolution layers.
- Added dropout before the final classifier.
- Kept the same input/output interface as V1:

```text
Input:  (batch_size, 1, 240)
Output: (batch_size, 4)
```

- Refactored training so both CNN models can be trained using the same script.
- Added `--model-name` support for selecting:
  - `cnn_baseline_v1`
  - `cnn_baseline_v2`

Example commands:

```bash
python -m ecg_arrhythmia.training.cnn_training --model-name cnn_baseline_v1
python -m ecg_arrhythmia.training.cnn_training --model-name cnn_baseline_v2
```

Best validation macro F1:

| Model | Best validation macro F1 |
|---|---:|
| CNN Baseline V1 | 0.4892 |
| CNN Baseline V2 | 0.4919 |

Key lesson:

- Keeping the same model input/output contract made it easy to reuse the same training loop.
- Refactoring the training script avoided duplicating hundreds of lines of code.
- BatchNorm and dropout improved the model without making the architecture too large for future edge deployment.

## Milestone 9 — Shared CNN evaluation and CNN Baseline V2 test results

Implemented:

- Refactored CNN evaluation so both V1 and V2 can be evaluated using the same script.
- Added model-specific checkpoint loading.
- Added model-specific metrics output paths.
- Evaluated both CNN baselines on the held-out test set.

Example commands:

```bash
python -m ecg_arrhythmia.evaluation.evaluate_cnn --model-name cnn_baseline_v1
python -m ecg_arrhythmia.evaluation.evaluate_cnn --model-name cnn_baseline_v2
```

Saved outputs:

```text
artifacts/results/cnn_baseline_v1_test_metrics.json
artifacts/results/cnn_baseline_v2_test_metrics.json
```

Test results:

| Model | Test loss | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| CNN Baseline V1 | 1.2953 | 0.7177 | 0.2807 |
| CNN Baseline V2 | 1.0929 | 0.7121 | 0.3256 |

Per-class test F1:

| Class | CNN V1 F1 | CNN V2 F1 | Change |
|---|---:|---:|---:|
| N | 0.8372 | 0.8268 | -0.0104 |
| S | 0.0287 | 0.0537 | +0.0250 |
| V | 0.2450 | 0.4118 | +0.1668 |
| F | 0.0120 | 0.0100 | -0.0020 |

Key lesson:

- CNN V2 is the stronger CNN baseline overall.
- Test accuracy decreased slightly, but macro F1 improved from `0.2807` to `0.3256`.
- This is a better trade-off for an imbalanced arrhythmia classification task.
- The largest improvement came from class `V`, where F1 increased from `0.2450` to `0.4118`.
- `S` improved slightly but remains weak.
- `F` remains extremely poor and difficult to judge because the test set contains very few `F` examples.

## Milestone 10 — Per-beat normalisation experiment

Implemented:

- Added optional per-beat z-score normalisation during beat extraction.
- Each beat window can now be normalised using its own mean and standard deviation:

```text
normalised_beat = (beat - beat.mean()) / beat.std()
```

- Kept normalisation optional so the original raw-signal baseline remains reproducible.
- Saved normalised processed data separately from the original dataset.
- Created separate normalised train/validation/test splits.
- Updated training and evaluation scripts so models can be trained and evaluated using custom split directories and checkpoint paths.
- Trained and evaluated normalised versions of both CNN baselines.

saved outputs to 
```text
artifacts/models/cnn_baseline_v1_normalised.pt
artifacts/models/cnn_baseline_v2_normalised.pt
artifacts/results/cnn_baseline_v1_normalised_test_metrics.json
artifacts/results/cnn_baseline_v2_normalised_test_metrics.json
```

Test results:

| Model | Normalised? | Test loss | Test accuracy | Test macro F1 |
|:---:|:---:|:---:|:---:|:---:|
| CNN Baseline V1 | No | 1.2953 | 0.7177 | 0.2807 |
| CNN Baseline V1 | Yes | 0.9969 | 0.8684 | 0.3737 |
| CNN Baseline V2 | No | 1.0929 | 0.7121 | 0.3256 |
| CNN Baseline V2 | Yes | 1.1696 | 0.7349 | 0.3131 |

Per-class test F1:

| Class | V1 raw | V1 normalised | V2 raw | V2 normalised |
|:---:|:---:|:---:|:---:|:---:|
| N | 0.8372 | 0.9298 | 0.8268 | 0.8406 |
| S | 0.0287 | 0.0049 | 0.0537 | 0.0404 |
| V | 0.2450 | 0.5600 | 0.4118 | 0.3713 |
| F | 0.0120 | 0.0000 | 0.0100 | 0.0000 |

key lesson:
- Per-beat normalisation significantly improved CNN Baseline V1 overall.
- V1 macro F1 increased from 0.2807 to 0.3737.
- The largest improvement came from class V, where F1 increased from 0.2450 to 0.5600.
- Normalisation also improved V1 accuracy from 0.7177 to 0.8684.
- CNN Baseline V2 did not benefit from normalisation overall; macro F1 decreased from 0.3256 to 0.3131.
- S remains very weak across all experiments, suggesting that morphology-only beat windows are not enough for supraventricular beat detection. RR intervals can help here because supraventricular beats are often reflected more clearly in rhythm/timing patterns than in QRS morphology. They can occur prematurely with a shortened previous RR interval, while their QRS morphology may still look close to normal, which is why they are often predicted as N. So RR features could provide context that the CNN is currently missing.
- F remains difficult to interpret because the test set contains only a very small number of F beats.
- The next improvement should add rhythm/context information, such as RR interval features. Since the goal is real-time edge compute I'll add previous RR intervals only

## Milestone 11 — RR Interval Features and CNN Baseline V2 RR

### Implemented

Added RR interval features to the ECG preprocessing and CNN training pipeline.

Each beat-level sample can now include:

- the 240-sample ECG beat window
- RR timing features:
  - previous RR interval in seconds
  - RR ratio relative to recent rhythm

Updated the project so RR features flow through:

- beat extraction
- dataset building
- dataset loading and validation
- patient-wise train/validation/test splitting
- PyTorch `ECGDataset`
- CNN training
- CNN evaluation

Added a new model:

- `CNNBaselineV2RR`

This model combines ECG morphology features learned by the CNN with RR timing features before classification.

### Comparison Tested

To isolate the effect of RR intervals, the key comparison is:

| Model | RR Features? | Normalised? |
|:---:|:---:|:---:|
| CNN Baseline V2 | No | No |
| CNN Baseline V2 RR | Yes | No |

This comparison keeps normalisation fixed and only changes whether RR interval features are provided.

### Test Results

| Model | Test Loss | Test Accuracy | Test Macro F1 |
|:---:|:---:|:---:|:---:|
| CNN Baseline V2 | 1.0929 | 0.7121 | 0.3256 |
| CNN Baseline V2 + RR | 0.7311 | 0.8014 | 0.4113 |

### Per-Class F1

| Class | CNN V2 | CNN V2 + RR | Change |
|:---:|:---:|:---:|:---:|
| N | 0.8268 | 0.8936 | +0.0668 |
| S | 0.0537 | 0.2397 | +0.1860 |
| V | 0.4118 | 0.5097 | +0.0979 |
| F | 0.0100 | 0.0023 | -0.0077 |

### Key Findings

Adding RR interval features improved overall model performance.

Macro F1 increased from `0.3256` to `0.4113`, showing that the RR-enhanced model performed better across classes rather than only improving accuracy on the majority class.

The largest improvement was on the `S` class:

| Metric | CNN V2 | CNN V2 + RR |
|:---:|:---:|:---:|
| Precision | 0.0347 | 0.3096 |
| Recall | 0.1186 | 0.1955 |
| F1 | 0.0537 | 0.2397 |

This supports the original motivation outlined yesterday for adding RR intervals. Supraventricular ectopic beats can look morphologically similar to normal beats, so timing context helps the model identify them.

### Confusion Matrix Notes

For true `S` beats:

| Model | Predicted N | Predicted S | Predicted V | Predicted F |
|:---:|:---:|:---:|:---:|:---:|
| CNN V2 | 75 | 37 | 131 | 69 |
| CNN V2 + RR | 45 | 61 | 177 | 29 |

The RR model correctly identified more `S` beats, increasing correct `S` predictions from `37` to `61`.

It also reduced the number of `S` beats incorrectly predicted as `N` or `F`.

### Conclusion

`CNNBaselineV2RR` is the strongest CNN baseline so far.

The results show that adding RR interval features improves performance over the morphology-only CNN V2 baseline, especially for the minority `S` class.

Current best CNN model:

### Current Best Baseline

| Model | Test macro F1 |
|:---:|:---:|
| CNN V2 without RR | 0.3256 |
| CNN V2 + RR normalised | 0.3644 |
| CNN V1 + normalised | 0.3737 |
| CNN V2 + RR unnormalised | 0.4113 |

### Next steps

The next modelling stage should compare this RR-enhanced CNN against the transformer/sequence model, since the transformer should be able to use
surrounding beat context more naturally.

## Milestone 12 — Causal Beat Sequence Dataset for Transformer Inputs

### Implemented

Added `sequence_dataset.py` to begin the transformer/sequence-modelling stage.

This converts the existing beat-level dataset into causal K-beat sequences. For example, with `sequence_length = 5`, each sample is built as:

```text
[beat i-4, beat i-3, beat i-2, beat i-1, beat i] -> label for beat i
```

The sequence builder:
- creates sliding beat sequences within each ECG record
- prevents sequences from crossing record boundaries
- keeps the target label as the label of the final beat in the sequence
- carries RR features forward for every beat in the sequence
- saves target indices so each sequence can be traced back to the original beat

The resulting sequence arrays are shaped for later transformer use:

```text
X_sequences: (num_sequences, sequence_length, 240)
```
```text
rr_features_sequences: (num_sequences, sequence_length, 2)
```
```text
y: (num_sequences,)
```

### Key Lesson
The transformer model should not be trained on isolated beats like the CNN baselines. Instead, each sample should contain a short causal history of recent beats.

This casual beat sequence is important because for real-time inference, we will not have access to future beats.

This was created to prepare the project for the next modelling stage of comparing the current best CNN baseline, CNN V2 + RR, against a transformer-style sequence model.

## Milestone 13 — Patient-Level Splitting for Sequence Dataset

Implemented:

- Added `split_sequence_dataset.py` for splitting the causal beat sequence dataset.
- Created train, validation, and test splits for transformer-ready sequence data.
- Kept the split patient-level so no patient appears in more than one split.
- Loaded and split the sequence dataset arrays:
  - `X.npy`
  - `rr_features.npy`
  - `y.npy`
  - `patient_ids.npy`
  - `target_indices.npy`
- Used `sequence_segments.json` to group sequence rows by patient.
- Used Monte Carlo sampling to search for a split that approximately preserves:
  - target train/validation/test ratios
  - class distribution across splits
- Saved each split separately under `data/splits_sequences/` with:
  - `X.npy`
  - `rr_features.npy`
  - `y.npy`
  - `patient_ids.npy`
  - `target_indices.npy`
- Saved split metadata to:
  - `split_indices.npz`
  - `split_summary_metrics.json`
- Precomputed patient-level sequence counts and label counts to make the split search much faster.

Key lesson:

- The transformer dataset needs its own split because the samples are now K-beat sequences rather than individual beats.
- Patient-level splitting is still essential because sequence samples from the same patient could otherwise leak across train, validation, and test sets.
- The split cannot be perfectly balanced because patients have different numbers of sequences and different arrhythmia distributions.
- Precomputing patient-level statistics makes Monte Carlo split search much more efficient because each trial can score patients without repeatedly slicing large arrays.
- This is the final main data preparation step needed before training the transformer model.

## Milestone 14 — CNN + RR Transformer Model Forward Pass

Implemented:

- Added `CNNBeatEncoder` for encoding individual ECG beat windows.
- The CNN beat encoder converts each beat from:

```text
(batch_size, 1, 240)
```

into a fixed-size morphology embedding:

```text
(batch_size, 128)
```

- Added unit tests to confirm the CNN encoder:
  - returns the expected embedding shape
  - supports flattened K-beat batches shaped as `(batch_size * sequence_length, 1, 240)`

- Added `RRFeatureEncoder` for encoding RR timing features.
- The RR encoder converts RR features from:

```text
(batch_size, sequence_length, 2)
```

into RR embeddings:

```text
(batch_size, sequence_length, 16)
```

- Confirmed that `nn.Linear` can handle both single-beat RR inputs and K-beat RR sequence inputs because it applies to the final tensor dimension.

- Added `ECGSequenceTransformer`, the first combined sequence model.
- The model combines:
  - CNN ECG morphology embeddings
  - RR timing embeddings
  - learned positional embeddings
  - a Transformer encoder
  - a final classifier

The model forward pass now supports:

```text
X_seq:  (batch_size, sequence_length, 1, 240)
RR_seq: (batch_size, sequence_length, 2)
Output: (batch_size, 4)
```

- Added unit tests to confirm the transformer:
  - returns logits with shape `(batch_size, num_classes)`
  - supports different sequence lengths up to `max_sequence_length`
  - rejects zero-length sequences

Key lesson:

- The CNN encoder should process each beat independently, while the Transformer handles relationships across the K-beat sequence.
- Flattening `(batch_size, sequence_length, 1, 240)` into `(batch_size * sequence_length, 1, 240)` lets the same CNN encoder be reused for every beat.
- an MLP can be used with both sequences and single values since it only takes the last dimension into account. 
- Concatenating ECG morphology embeddings and RR timing embeddings gives each beat a combined representation before sequence modelling.
- The Transformer output for the final beat is used for classification because each causal sequence predicts the label of its final beat.

## Milestone 15 — Transformer Training, Tuning and Test Evaluation

Implemented:

- Added a dedicated training pipeline for `ECGSequenceTransformer`.
- Loaded causal sequence data using `ECGSequenceDataset`.
- Trained the model using:
  - ECG beat sequences shaped as `(batch_size, sequence_length, 1, 240)`
  - RR feature sequences shaped as `(batch_size, sequence_length, 2)`
  - the final beat label as the target
- Added weighted `CrossEntropyLoss` using class counts from the training sequence split.
- Used validation macro F1 to select and save the best checkpoint.
- Used scikit-learn's `classification_report` and `confusion_matrix` for:
  - accuracy
  - macro F1
  - per-class precision
  - per-class recall
  - per-class F1
  - confusion matrix
- Added early stopping with a patience of 10 epochs.
- Added a fixed random seed for reproducible experiments.
- Added configurable training options for:
  - learning rate
  - dropout
  - number of Transformer layers
  - class-weighting method
  - random seed
  - model output path

### Matched Patient Split

Rebuilt the sequence train, validation, and test sets using the same patient assignments as the CNN splits.

This ensured that the final Transformer and CNN comparison used:

- the same training patients
- the same validation patients
- the same test patients

The Transformer test set contains 28 fewer targets because a sequence length of 5 requires four previous beats for each record.

### Hyperparameter Experiments

| Experiment | Learning Rate | Dropout | Layers | Class Weighting | Best Validation Macro F1 |
|:---:|:---:|:---:|:---:|:---:|---:|
| Baseline | `1e-3` | `0.3` | 2 | inverse | 0.5940 |
| A | `3e-4` | `0.3` | 2 | inverse | 0.6466 |
| B | `1e-4` | `0.3` | 2 | inverse | 0.6204 |
| C | `3e-4` | `0.1` | 2 | inverse | 0.6546 |
| D | `3e-4` | `0.2` | 2 | inverse | 0.6624 |
| E | `3e-4` | `0.2` | 1 | inverse | 0.6160 |
| F | `3e-4` | `0.2` | 3 | inverse | **0.6901** |
| G | `3e-4` | `0.2` | 4 | inverse | 0.6750 |
| H | `3e-4` | `0.2` | 3 | square-root inverse | 0.6806 |
| I | `3e-4` | `0.2` | 3 | capped inverse | 0.6714 |

### Selected Transformer Configuration

The strongest validation configuration was Experiment F:

```text
Learning rate: 0.0003
Dropout: 0.2
Transformer layers: 3
Class weighting: inverse frequency
Seed: 42
Best validation macro F1: 0.6901
```

### Final Matched Test Results

Saved output:

```text
artifacts/results/ecg_sequence_transformer_tuned_matched_test_metrics.json
```

| Metric | Value |
|:---:|---:|
| Test loss | 0.4662 |
| Test accuracy | 0.9496 |
| Test macro F1 | 0.5340 |

Per-class test F1:

| Class | F1 |
|:---:|---:|
| N | 0.9766 |
| S | 0.2939 |
| V | 0.8382 |
| F | 0.0274 |

### Key Lesson

- Validation data should be used for checkpoint selection and hyperparameter tuning.
- The test set should only be evaluated after the final configuration has been selected.
- Reducing the learning rate from `1e-3` to `3e-4`, lowering dropout to `0.2`, and increasing the Transformer depth to 3 layers produced the strongest validation result.
- Sequence context gave much stronger performance for `V` beats and improved `S` compared with the earlier matched Transformer baseline.
- `F` remains difficult because it has extremely low support and its metrics are highly unstable.

## Milestone 16 — Final Visualisations and ONNX Export

Implemented:

- Added final visualisations for the tuned Transformer and CNN V2 + RR baseline.
- Plotted the Transformer hyperparameter search and highlighted Experiment F as the best configuration.
- Added target-matched comparison plots for:
  - test accuracy
  - test macro F1
  - per-class F1
- Added row-normalised confusion matrices showing both raw counts and percentages.
- Exported the tuned three-layer Transformer from PyTorch to ONNX.
- Kept the sequence length fixed at 5 beats while allowing a dynamic batch size.

Saved figures:

```text
artifacts/figures/transformer_hyperparameter_search.png
artifacts/figures/cnn_vs_transformer_overall_metrics.png
artifacts/figures/cnn_vs_transformer_per_class_f1.png
artifacts/figures/transformer_tuned_confusion_matrix.png
artifacts/figures/cnn_v2_rr_target_matched_confusion_matrix.png
```

Final target-matched results:

| Model | Test accuracy | Test macro F1 |
|:---|---:|---:|
| CNN V2 + RR | 0.8012 | 0.4112 |
| Tuned Transformer | 0.9496 | 0.5340 |

Per-class test F1:

| Class | CNN V2 + RR | Tuned Transformer |
|:---:|---:|---:|
| N | 0.8935 | 0.9766 |
| S | 0.2397 | 0.2939 |
| V | 0.5091 | 0.8382 |
| F | 0.0023 | 0.0274 |

Saved ONNX model locally:

```text
artifacts/models/ecg_sequence_transformer.onnx
```

Key lesson:

- The tuned Transformer clearly outperformed the CNN V2 + RR baseline, especially for class `V`.
- Target-matched evaluation ensures both models are compared on the same test beats.
- Confusion matrices reveal class-specific behaviour that overall metrics can hide.
- ONNX export packages the trained model as a portable computational graph for later inference, parity testing, quantisation and edge deployment.

## Milestone 17 — PyTorch–ONNX Parity Verification

Implemented:

- Added a deployment validation script to compare the tuned PyTorch Transformer with its exported ONNX model.
- Validated that the ONNX graph is structurally correct and exposes the expected ECG, RR and logits nodes.
- Ran the complete matched test set through both PyTorch and ONNX Runtime using identical inputs.
- Compared the raw logits using configurable numerical tolerances.
- Confirmed that both implementations produced matching final class predictions.
- Added a failure condition when either logit parity or prediction parity is not violated.

Saved output:

```text
artifacts/results/deployment/pytorch_onnx_parity_summary.json
```

Key lesson:

- Successfully exporting a model does not guarantee that its behaviour has been preserved.
- The parity check confirms both models have numerical equivalent logits and agree on the final predictions.
- The exported ONNX model now provides a validated deployment representation of the tuned PyTorch Transformer.

## Milestone 18 — R-Peak Matching and Detector Validation

### Implemented

- Added a shared `RPeakDetector` interface with validation for ECG signals, sampling rates, and returned peak indices.
- Implemented and tested three interchangeable detectors:
  - XQRS
  - Hamilton
  - Elgendi
- Added chronological one-to-one matching between expert annotations and detected R-peaks.
- Enforced that each annotation can match only one detection and each detection can match only one annotation.
- Defined the signed timing offset as:

```text
detected sample - expert annotation sample
```

- Added validation metrics for:
  - true positives, false positives, and false negatives;
  - precision, recall, and F1;
  - mean and median timing offset;
  - mean and maximum absolute timing error;
  - detector runtime and real-time speed;
  - per-record and per-symbol performance.
- Added `evaluate_r_peak_validation.py` to evaluate all detectors consistently on the validation records.

### Validation Results

| Detector | F1 | False Positives | False Negatives | Mean Absolute Offset |
|:---:|:---:|:---:|:---:|:---:|
| XQRS | **0.9972** | **5** | **77** | **1.97 ms** |
| Hamilton | 0.9961 | 13 | 102 | 53.47 ms |
| Elgendi | 0.9791 | 521 | 101 | 71.21 ms |

XQRS was selected because it achieved the best overall detection quality and substantially better peak-localisation accuracy while remaining fast enough for real-time inference.

### Saved Outputs

```text
artifacts/results/detection_evaluation/xqrs_metrics.json
```

### Key Lesson

A high detector F1 alone is not enough for this project. Accurate R-peak localisation is also important because timing errors shift the ECG windows and RR features supplied to the transformer. XQRS provided the best balance of recall, precision, and localisation accuracy.

## Milestone 19 — XQRS-Centred Dataset and Paired Validation Comparison

### Implemented

- Added an XQRS-centred sequence dataset builder for deployment-style model inputs.
- Centred ECG windows on detected R-peaks rather than expert annotation locations.
- Calculated RR features from the full detected R-peak timeline.
- Allowed unmatched detections to influence RR features and sequence context.
- Used only matched detections with supported AAMI labels as scored targets.
- Added audit arrays for:
  - record identity;
  - detected and expert annotation samples;
  - timing offsets in samples and milliseconds;
  - unmatched detections within sequence context.
- Rebuilt the expert-centred validation sequences while retaining stable target identities.
- Matched expert-centred and XQRS-centred targets using:

```python
(record_name, expert_annotation_sample)
```

- Created aligned paired dataset views containing the same targets in the same order.
- Evaluated the original tuned transformer checkpoint under both centring conditions.
- Added checks to confirm that target identities, labels, sequence counts, and prediction ordering remained aligned.
- Compared:
  - loss, accuracy, and macro F1;
  - per-class F1;
  - prediction agreement;
  - correct-to-incorrect and incorrect-to-correct transitions.

### Paired Validation Result

The paired comparison contained `14,548` shared validation targets.

| Metric | Expert-Centred | XQRS-Centred | Change |
|---|---:|---:|---:|
| Accuracy | **0.9687** | 0.9634 | -0.0053 |
| Macro F1 | **0.6920** | 0.6752 | -0.0168 |

Prediction agreement was `99.03%`.

The original transformer remained highly stable when expert-centred windows were replaced with XQRS-centred windows. The small reduction in performance established the deployment gap before any XQRS-specific fine-tuning was applied.

### Saved Outputs

```text
data/splits_sequences_xqrs/val/

data/splits_sequences_paired/expert_centered/
data/splits_sequences_paired/xqrs_centered/

artifacts/results/model_evaluation/transformer_paired_centering_comparison.json
```

### Key Lesson

A detector can achieve excellent R-peak metrics while still changing the exact ECG morphology and RR context seen by the classifier. Pairing the same targets across expert-centred and XQRS-centred inputs provided a controlled measurement of this effect. XQRS caused only a small reduction in performance, confirming that it was suitable for the deployment pipeline while providing a baseline for later fine-tuning.

## Milestone 20 — XQRS-Centred Fine-Tuning and Final Test Evaluation

### Implemented

- Extended `transformer_training.py` to support optional checkpoint initialisation through `--initial-checkpoint-path`.
- Loaded initial weights with `strict=True` before creating the optimiser.
- Evaluated the loaded checkpoint on the XQRS-centred validation set before epoch 1.
- Saved the baseline checkpoint as the initial best model so that a worse fine-tuning epoch could not replace it.
- Saved a compact fine-tuning summary containing:
  - configuration;
  - completed epochs and best epoch;
  - baseline metrics;
  - fine-tuned metrics;
  - absolute overall and per-class changes.
- Built an XQRS-centred test split from the held-out test records.
- Added `evaluate_xqrs_test_checkpoints.py` to evaluate the original and EXPC checkpoints on the exact same test sequences.
- Recorded:
  - loss, accuracy, macro F1 and per-class metrics;
  - confusion matrices;
  - prediction agreement;
  - correctness transitions;
  - EXPC-minus-original metric changes.

### Selected Fine-Tuning Configuration

The validation-best configuration was saved as the EXPC checkpoint:

```text
Initial checkpoint:  artifacts/models/ecg_sequence_transformer_tuned.pt
Output checkpoint:   artifacts/models/ecg_sequence_transformer_xqrs_EXPC.pt

Transformer layers:  3
Dropout:              0.3
Learning rate:        7e-6
Batch size:           64
Maximum epochs:       40
Patience:             15
Seed:                 42
Class weighting:      capped inverse
Maximum class weight: 5
```

Training stopped early after 19 completed epochs, with the best checkpoint selected at epoch 4.

### XQRS-Centred Validation Result

| Metric | Original | EXPC | Change |
|:---:|:---:|:---:|:---:|
| Loss | 0.2119 | **0.1952** | -0.0166 |
| Accuracy | 0.9634 | **0.9791** | +0.0157 |
| Macro F1 | 0.6752 | **0.6872** | +0.0120 |
| N F1 | 0.9806 | **0.9893** | +0.0087 |
| S F1 | 0.7925 | **0.8146** | +0.0221 |
| V F1 | 0.9276 | **0.9449** | +0.0173 |
| F F1 | 0.0000 | 0.0000 | 0.0000 |

The EXPC checkpoint improved validation accuracy, macro F1, and the N, S and V class F1 scores.

### Final XQRS-Centred Test Set

The locked test dataset contained seven unseen records:

```text
100, 103, 118, 121, 207, 221, 223
```

```text
Final sequences: 15,263

Class distribution:
N: 13,923
S:    312
V:  1,014
F:     14
```

### Final Locked Test Comparison

| Metric | Original | EXPC | Change |
|:---:|:---:|:---:|:---:|
| Loss | **0.4119** | 0.4241 | +0.0122 |
| Accuracy | 0.9465 | **0.9518** | +0.0052 |
| Macro F1 | **0.5268** | 0.5001 | -0.0267 |
| N F1 | 0.9747 | **0.9799** | +0.0052 |
| S F1 | **0.2782** | 0.2547 | -0.0235 |
| V F1 | **0.8235** | 0.7658 | -0.0577 |
| F F1 | **0.0308** | 0.0000 | -0.0308 |

Prediction comparison:

```text
Targets:                 15,263
Identical predictions:   14,813
Prediction agreement:    97.05%
Changed predictions:        450

Correct → correct:       14,311
Correct → incorrect:        136
Incorrect → correct:        216
Incorrect → incorrect:      600
```

EXPC corrected 216 predictions that the original model got wrong, while changing 136 originally correct predictions into errors. This produced a net increase in accuracy.

However, the additional correct predictions were concentrated in the dominant N class. EXPC reduced S, V and F performance, causing test macro F1 to fall.

### Final Decision

The validation improvement from XQRS-centred fine-tuning did not generalise to the held-out test patients.

The original tuned checkpoint was retained as the final deployment model:

```text
artifacts/models/ecg_sequence_transformer_tuned.pt
```

The EXPC checkpoint was preserved as the validation-selected fine-tuning candidate, but it was not selected for deployment because the original checkpoint achieved:

- higher test macro F1;
- stronger S performance;
- substantially stronger V performance;
- non-zero F performance;
- lower test loss.

### Saved Outputs

```text
data/splits_sequences_xqrs/train/
data/splits_sequences_xqrs/val/
data/splits_sequences_xqrs/test/

artifacts/models/ecg_sequence_transformer_xqrs_EXPC.pt

artifacts/results/model_evaluation/transformer_xqrs_EXPC_summary.json
artifacts/results/model_evaluation/transformer_xqrs_test_comparison.json
```

### Key Lesson

Matching the training distribution to deployment inputs can improve validation performance, but the improvement must be confirmed on unseen patients.

XQRS-centred fine-tuning increased validation macro F1 and final test accuracy, but reduced final test macro F1 by weakening the minority arrhythmia classes. This concludes this portion of the project of evaluating the detectors. This has told me that the original tuned model on expert annotated beats is robust enough to perform well on detected beats. This has also allowed me to select the best detector out of xqrs, elgendi, and hamilton to use for the upcoming real time inference portion of the project.   

## Milestone 21 — Real-Time Streaming Replay Foundation

### Implemented

Added the first stage of the real-time ECG streaming pipeline.

Created:

- `SampleChunk` for immutable, validated ECG sample blocks;
- `ReplaySource` for yielding records as consecutive chunks;
- accelerated and real-time replay modes;
- `StreamingEngine` for validating stream continuity;
- `StreamState` for tracking the current record;
- a replay command-line tool with JSON summary output.

The replay source preserves sample order and absolute indices, includes final partial chunks, and uses absolute monotonic-clock targets in real-time mode to avoid accumulated timing drift.

The streaming engine rejects:

- missing samples;
- duplicate chunks;
- overlapping or out-of-order chunks;
- sampling-rate changes within a record.

Added unit and integration tests for chunk validation, sample preservation, replay timing, continuity checks, state resets and real MIT-BIH replay.

### Validation Results

Accelerated replay of record `114`:

| Metric | Result |
|---|---:|
| Samples accepted | 650,000 |
| Chunks emitted | 18,056 |
| Elapsed time | 0.1032 s |
| Continuity valid | True |

Ten-second real-time replay of record `114`:

| Metric | Result |
|---|---:|
| Samples accepted | 3,600 |
| Chunks emitted | 100 |
| Elapsed time | 10.0002 s |
| Continuity valid | True |

Saved outputs:

```text
artifacts/results/streaming_evaluation/record_114_accelerated_replay_summary.json
artifacts/results/streaming_evaluation/record_114_real_time_10s_replay_summary.json
```

### Key Lesson

A real-time inference system first needs a reliable sample-transport layer. Separating the replay source, chunk representation and streaming engine allows MIT-BIH replay to later be replaced by a Raspberry Pi or live ECG source without changing the downstream interface.

## Milestone 22 — Streaming XQRS Beat-Sequence Pipeline

### Implemented

Completed Section 2 of the real-time streaming pipeline. Incoming ECG chunks can now be converted into model-ready, XQRS-centred causal sequences.

The streaming flow is now:

```text
SampleChunk
    → rolling ECG buffer
    → causal overlapping-window XQRS
    → confirmed R-peaks
    → 240-sample beat windows
    → previous-RR and RR-ratio features
    → five-beat causal sequences
```

Each emitted sequence contains:

- ECG input with shape `(5, 1, 240)`;
- RR input with shape `(5, 2)`;
- the absolute target R-peak index.

Added separate components for:

- absolute-indexed rolling sample storage
- streaming XQRS detection with overlap, warm-up and confirmation
- delayed beat construction once post-peak samples are available
- shared offline/streaming RR-feature calculation
- sliding five-beat sequence construction
- per-record and all-validation parity evaluation
- diagnostic reporting and plots for detector divergences.

The observed streaming behaviour was also confirmed to be consistent across several chunk sizes.

No model inference is performed yet.

### Validation Results

Evaluated all six validation records:

```text
114, 122, 209, 210, 231, 233
```

| Metric | Result |
|:---:|:---:|
| Samples accepted | 3,900,000 / 3,900,000 |
| Continuity validated | True for every record |
| Whole-record XQRS peaks | 14,588 |
| Exactly matched peaks | 14,587 |
| Missing streaming peaks | 1 |
| Extra streaming peaks | 4 |
| Maximum shared-peak offset | 0 samples |

Sequence parity results:

| Metric | Result |
|:---:|:---:|
| Offline targets | 14,548 |
| Exactly matched sequences | 14,496 |
| Expected deployment-only targets | 5 |
| Causal-detector-only targets | 4 |
| Missing due to causal divergence | 1 |
| Unexplained extra targets | 0 |
| Unexplained missing targets | 0 |
| Unexplained content mismatches | 0 |

All 18 ECG-window mismatches and all 51 RR-feature mismatches were explained by the small number of causal-versus-whole-record XQRS detection differences.

The final aggregate reports:

```text
all_records_exact_parity:          False
all_records_differences_explained: True
```

Exact parity is false because the causal detector and whole-record detector make a few different decisions, but no unexplained streaming-preprocessing defect was found.

### Important Dataset Caveat

Parity is measured against:

```text
data/splits_sequences_xqrs/val
```

This is an XQRS-centred dataset, but it is still expert-influenced. All XQRS detections can affect RR history and sequence context, while only detections matched to supported expert annotations are retained as labelled target sequences.

This explains record `122`, where streaming reproduced every offline target exactly but emitted one additional valid sequence that was excluded from the expert-filtered validation set.

### Key Lesson

The streaming engine now reproduces the validated offline preprocessing behaviour wherever the detector timelines agree.
The important result is that every remaining peak, ECG-window and RR-feature difference was localised and explained, with no unexplained streaming-preprocessing defect found.

## Milestone 23 — Streaming ONNX Inference and Three-Way Parity

### Implemented

Completed Section 3 of the real-time streaming pipeline. Model-ready `BeatSequence` objects emitted by the streaming engine can now be classified directly with ONNX Runtime.

The streaming flow is now:

```text
SampleChunk
    → StreamingEngine
    → BeatSequence
    → StreamingPredictor
    → ONNXSequenceClassifier
    → PredictionEvent
```

Added `onnx_contract.py` to define and validate the deployed ONNX interface:

- ECG input name: `ecg_sequence`
- RR input name: `rr_sequence`
- output name: `logits`
- ECG input rank: 4
- RR input rank: 3
- logits output rank: 2
- expected float32 input types
- expected sequence length, ECG window size, RR-feature dimension and number of output classes
- CPU ONNX Runtime session creation
- structural ONNX model validation.

The `onnx` package is imported only when structural model validation is requested, while the normal inference path uses `onnxruntime`.

Added `ONNXSequenceClassifier` to:

- accept one streaming `BeatSequence`;
- validate the ECG input shape `(5, 1, 240)`;
- validate the RR input shape `(5, 2)`;
- convert both inputs to contiguous `float32`;
- add a batch dimension:
  - ECG: `(5, 1, 240) -> (1, 5, 1, 240)`
  - RR: `(5, 2) -> (1, 5, 2)`
- run the sequence through the ONNX Runtime session;
- validate the returned logits shape `(1, 4)`;
- reduce the output to a read-only `(4,)` logit vector;
- convert the largest logit into the predicted class index and AAMI label.

Added immutable `PredictionEvent` outputs containing:

- target R-peak index;
- all R-peaks that formed the five-beat sequence;
- raw model logits;
- predicted class index;
- predicted class label.

Added `StreamingPredictor` as a composition layer over `StreamingEngine`.

`StreamingPredictor`:

- forwards each `SampleChunk` through the existing streaming engine;
- classifies every `BeatSequence` emitted by that chunk;
- classifies any final sequences released by `flush()`;
- returns only the predictions produced by the current call;
- retains no prediction history between records.

Added a three-way inference-parity evaluator using the exact same streaming-emitted `BeatSequence` inputs for:

```text
PyTorch
Offline ONNX
Streaming ONNX
```

The three comparisons are:

```text
PyTorch vs Offline ONNX
Offline ONNX vs Streaming ONNX
PyTorch vs Streaming ONNX
```

The streaming classifier records each exact `BeatSequence` that produced a live prediction. Those same sequence objects are then passed through the original PyTorch model and through a second independently created ONNX session, preventing input differences from affecting the parity test.

Added per-sequence comparison of:

- predicted class agreement
- mean absolute logit difference
- maximum absolute logit difference
- exact array equality
- `np.allclose` tolerance checks using:
  - `rtol = 1e-5`
  - `atol = 1e-5`
- target R-peaks for any class disagreements or tolerance failures
- class-agreement matrices.

Added optional parity figures:

- row-normalised class-agreement matrices;
- PyTorch-vs-streaming-ONNX logit scatter plots;
- per-sequence maximum-logit-difference histograms;
- logit-difference plots across each ECG record.

Parity results and raw logits are written under:

```text
artifacts/results/deployment_evaluation/streaming_inference_parity/
```

Figures are written under:

```text
artifacts/figures/streaming_inference_parity/
```

### Validation Results

Evaluated all six validation records:

```text
114, 122, 209, 210, 231, 233
```

using a streaming chunk size of 36 samples.

A total of **14,556 streaming-emitted sequences** were compared.

| Comparison | Class agreement | Mean absolute logit difference | Maximum absolute logit difference | Arrays outside tolerance | Exactly equal |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PyTorch vs Offline ONNX | 100% | 5.665e-07 | 9.418e-06 | 0 | False |
| Offline ONNX vs Streaming ONNX | 100% | 0.000e+00 | 0.000e+00 | 0 | True |
| PyTorch vs Streaming ONNX | 100% | 5.665e-07 | 9.418e-06 | 0 | False |

All **14,556 / 14,556** sequences produced the same predicted class in all three inference paths.

All PyTorch-versus-ONNX logit arrays were within the required `rtol=1e-5`, `atol=1e-5` tolerance.

Offline ONNX and streaming ONNX were bit-for-bit identical across every sequence:

```text
mean absolute logit difference:    0.0
maximum absolute logit difference: 0.0
all arrays exactly equal:          True
```

The aggregate class-agreement matrix was fully diagonal:

| Prediction | N | S | V | F |
|:---:|:---:|:---:|:---:|:---:|
| N | 12,836 | 0 | 0 | 0 |
| S | 0 | 386 | 0 | 0 |
| V | 0 | 0 | 1,114 | 0 |
| F | 0 | 0 | 0 | 220 |

These counts represent agreement between inference implementations, not accuracy against expert ground-truth labels.

The final aggregate reports:

```text
failed_records:              []
records_failing_parity:      []
all_records_parity_passed:   True
```

### Key Lesson

Section 3 confirms that model inference can be attached to the real-time streaming pipeline without changing the model's behaviour.

The exact equality between offline ONNX and streaming ONNX shows that the streaming inference wrapper does not alter the model inputs or outputs. The very small PyTorch-versus-ONNX numerical differences are within the required tolerance and never change the predicted class.

The real-time pipeline now extends from raw ECG chunks through causal R-peak detection and beat-sequence assembly to a traceable `PredictionEvent`, while retaining the behaviour of the original trained PyTorch model.

## Milestone 24 — FP32 ONNX Inference Benchmark

### Implemented

- Added `benchmark_onnx_inference.py` to establish a deployment baseline for the FP32 ONNX Transformer.
- Benchmarked all six validation records using model-ready sequences from the streaming pipeline.
- Added warm-up calls and measured:
  - inference latency;
  - throughput;
  - classifier initialisation time;
  - model size.
- Added unit tests for benchmark calculations, warm-up behaviour and inference timing.

### Results

| Metric | Result |
|:------:|:------:|
| Sequences | 14,556 |
| Mean latency | 0.658 ms |
| Median latency | 0.597 ms |
| p95 latency | 1.006 ms |
| Throughput | 1,519.85 sequences/s |
| Model size | 2.310 MiB |

Saved output:

```text
artifacts/results/deployment_evaluation/onnx_benchmarking/fp32_onnx_benchmark.json
```

### Key Lesson

The FP32 ONNX model is already very fast on the development machine. This benchmark provides the reference point needed to judge whether INT8 quantisation improves deployment performance.


## Milestone 25 — Dynamic INT8 ONNX Quantisation

### Implemented

- Added `quantize_onnx.py` to create a dynamically quantised QInt8 ONNX model.
- Added ONNX Runtime preprocessing with fallback when symbolic shape inference fails.
- Removed stale `graph.value_info` metadata before quantisation and revalidated the cleaned graph.
- Added structural validation, runtime contract validation and smoke inference for the INT8 model.
- Added unit tests for quantisation flow, metadata cleaning, fallbacks, validation and reporting.

### Results

| Model | Size |
|:-----:|:----:|
| FP32 | 2.310 MiB |
| INT8 | 0.813 MiB |

Model size reduction:

```text
64.79% ≈ 2.84× smaller
```

All structural, contract and smoke-inference checks passed.

Saved outputs:

```text
artifacts/results/deployment_evaluation/quantization/dynamic_int8_quantization_report.json
```

### Key Lesson

Dynamic quantisation reduced the model size substantially while preserving a valid, executable deployment graph. The next steps are to test FP32-vs-INT8 prediction agreement, classification performance and inference speed.

## Milestone 26 — FP32 vs INT8 Inference Agreement

### Implemented

- Added `evaluate_quantized_inference_agreement.py` to measure the numerical and prediction-level impact of dynamic INT8 quantisation.
- Evaluated the FP32 and INT8 ONNX models on the exact same real streaming-emitted `BeatSequence` inputs across all six validation records.
- Added comparison metrics for:
  - class agreements and disagreements;
  - class agreement percentage;
  - FP32-to-INT8 class transition counts;
  - mean and maximum absolute logit drift;
  - per-sequence maximum absolute logit drift;
  - FP32 and INT8 decision margins.
- Added traceable disagreement records containing the target peak, predicted labels, logits, logit differences and decision margins.
- Added correctly pooled aggregate statistics across records so records contribute proportionally to their sequence counts.
- Added `quantization_agreement_plots.py` for:
  - FP32-vs-INT8 agreement matrices;
  - FP32-vs-INT8 logit scatter plots;
  - per-sequence drift histograms;
  - drift across ECG records;
  - agreeing-vs-disagreeing FP32 margin comparisons;
  - aggregate agreement and drift figures.
- Added unit tests for agreement calculations, logit drift, margins, transition counts, input validation, identical sequence ordering, disagreement tracing, aggregate pooling and plot generation.

### Results

| Metric | Result |
| ------ | ------: |
| Validation records | 6 |
| Sequences compared | 14,556 |
| Class agreements | 14,461 |
| Class disagreements | 95 |
| Class agreement | 99.3473% |
| Class disagreement | 0.6527% |
| Mean absolute logit difference | 0.0721 |
| Maximum absolute logit difference | 1.2124 |
| Mean per-sequence maximum difference | 0.1409 |
| Median per-sequence maximum difference | 0.1229 |
| p95 per-sequence maximum difference | 0.3084 |

Record `122` achieved 100% FP32-vs-INT8 class agreement.

The 95 disagreements were distributed as:

```text
Record 114: 16
Record 122:  0
Record 209: 24
Record 210:  5
Record 231:  1
Record 233: 49
```

Disagreeing sequences consistently had much smaller FP32 decision margins than agreeing sequences, indicating that quantisation-induced class changes were concentrated around predictions already close to a decision boundary.

Saved outputs:

```text
artifacts/results/deployment_evaluation/quantization_agreement/
    quantization_agreement_summary.json
    fp32_vs_int8_disagreements.json
    record_<record>.json

artifacts/figures/quantization_agreement/
    aggregate_fp32_vs_int8_agreement.png
    aggregate_difference_histogram.png
    record_<record>_fp32_vs_int8_agreement.png
    record_<record>_logit_scatter.png
    record_<record>_difference_histogram.png
    record_<record>_difference_across_record.png
    record_<record>_margin_comparison.png
```

### Key Lesson

Dynamic INT8 quantisation preserved the FP32 model's predicted class on **99.35% of 14,556 identical streaming sequences** while introducing only minimal numerical logit drift.

The substantially smaller FP32 margins observed on disagreeing sequences indicate that quantisation primarily changed borderline decisions rather than predictions with strongly separated class logits.

This establishes that the INT8 model retains very high behavioural agreement with the FP32 deployment model. The next step is to compare both models against ground-truth labels to determine whether the remaining prediction changes improve or degrade classification performance.

## Milestone 27 — FP32 vs INT8 Ground-Truth Classification Performance

### Implemented

- Added `evaluate_quantized_model_performance.py` to determine whether INT8 quantisation changes actual classification quality relative to the FP32 deployment model.
- Evaluated the FP32 and INT8 ONNX models against the same supported ground-truth AAMI labels using identical streaming-emitted sequences.
- Added ground-truth alignment using detected target R-peaks so both models are scored on exactly the same labelled subset.
- Added classification metrics for:
  - accuracy;
  - macro F1;
  - per-class precision;
  - per-class recall;
  - per-class F1;
  - per-class support;
  - confusion matrices.
- Added signed INT8-minus-FP32 metric deltas for overall and per-class performance.
- Added analysis of changed predictions to determine whether each FP32-to-INT8 disagreement was:
  - FP32 correct and INT8 wrong;
  - FP32 wrong and INT8 correct;
  - wrong under both models.
- Added per-ground-truth-class changed-prediction counts and a net correct-prediction change.
- Recomputed aggregate metrics from pooled labelled predictions across all validation records rather than averaging per-record scores.
- Added `quantized_model_performance_plots.py` for:
  - FP32 ground-truth confusion matrix;
  - INT8 ground-truth confusion matrix;
  - per-class F1 comparison;
  - per-class recall comparison;
  - INT8-minus-FP32 metric deltas;
  - changed-prediction outcome counts.
- Added unit tests for ground-truth alignment, classification metrics, metric deltas, changed-prediction outcomes, aggregate pooling and plot generation.

### Results

| **Metric** | **FP32** | **INT8** | **Change** |
|---|---:|---:|---:|
| Accuracy | 0.96357 | **0.96563** | **+0.00206** |
| Macro F1 | 0.67577 | **0.69014** | **+0.01437** |
| N F1 | 0.98065 | **0.98176** | +0.00111 |
| S F1 | 0.79353 | **0.83895** | **+0.04542** |
| V F1 | 0.92891 | **0.93207** | +0.00316 |
| F F1 | 0.00000 | **0.00778** | +0.00778 |

INT8 therefore did not introduce a classification-performance penalty on the validation data.

Despite the small number of class changes identified during the FP32-vs-INT8 agreement evaluation, the quantised model produced slightly higher accuracy and macro F1.

The largest improvement was on the minority `S` class, where F1 increased from `0.79353` to `0.83895`.

### Saved Outputs

```text
artifacts/results/deployment_evaluation/quantized_model_performance/
    quantized_model_performance_summary.json
    quantized_model_performance_arrays.npz

artifacts/figures/quantized_model_performance/
    fp32_confusion_matrix.png
    int8_confusion_matrix.png
    per_class_f1_comparison.png
    per_class_recall_comparison.png
    metric_deltas.png
    changed_prediction_outcomes.png
```

### Key Lesson

Dynamic INT8 quantisation preserved the classification quality of the FP32 deployment model and produced a small improvement on the validation data.

The increase in macro F1 from `0.67577` to `0.69014`, together with improvements across all four per-class F1 scores, shows that the prediction changes introduced by quantisation did not degrade ground-truth performance.

This establishes that the substantially smaller INT8 model retains the predictive quality required for deployment evaluation. The next step is to benchmark FP32 and INT8 inference performance under identical conditions to determine whether quantisation also provides a computational advantage.

## Milestone 28 — FP32 vs INT8 Inference Performance Benchmark

### Implemented

- Added `benchmark_quantized_inference.py` to directly compare the computational performance of the FP32 and dynamically quantised INT8 ONNX models.
- Benchmarked both models using the exact same streaming-emitted `BeatSequence` inputs across all six validation records.
- Reused the same ONNX Runtime CPU execution provider and identical session configuration for both models.
- Added 100 warm-up inference calls before each timed pass so the benchmark measures steady-state inference rather than first-call setup effects such as lazy initialisation and buffer allocation.
- Added five repeated full benchmark passes for each model rather than relying on a single timing run.
- Added deterministic counterbalancing of model execution order:

```text
FP32 -> INT8
INT8 -> FP32
FP32 -> INT8
INT8 -> FP32
FP32 -> INT8
```

- Alternating which model runs first reduces systematic bias from machine-load, CPU-state and thermal drift during the benchmark.
- Recorded per-repeat:
  - minimum latency;
  - mean latency;
  - median latency;
  - p95 latency;
  - maximum latency;
  - throughput.
- Added across-repeat summaries to measure run-to-run variation.
- Added pooled latency statistics across every timed prediction.
- Timed FP32 and INT8 classifier initialisation separately from steady-state model inference.
- Added direct INT8-minus-FP32 comparisons for:
  - mean latency;
  - median latency;
  - p95 latency;
  - throughput;
  - classifier initialisation time.
- Added latency and throughput percentage changes and speedup ratios.
- Added direct model-size comparison including:
  - FP32 and INT8 size in bytes and MiB;
  - absolute size reduction;
  - percentage reduction;
  - compression ratio.
- Preserved record boundaries within the pooled sequence collection so FP32 and INT8 latency could also be compared separately for each validation record.
- Added `onnx_benchmark_plots.py` for:
  - FP32-vs-INT8 latency comparison;
  - FP32-vs-INT8 throughput comparison;
  - model-size comparison;
  - per-record mean-latency comparison.
- Added unit tests for counterbalanced ordering, repeated benchmarking, warm-up behaviour, summary statistics, comparison metrics, model-size calculations, per-record pooling and plot generation.

### Results

The controlled benchmark used all six validation records and the same `14,556` streaming-emitted sequences for both models, with five full timed passes per model and 100 warm-up calls before each pass.

| **Metric** | **FP32** | **INT8** | **INT8 vs FP32** |
|---|---:|---:|---:|
| Mean latency | 0.681 ms | 3.658 ms | +436.79% |
| Median latency | 0.618 ms | 3.493 ms | +465.25% |
| p95 latency | 1.013 ms | 4.447 ms | +339.15% |
| Throughput | 1,473.46 sequences/s | 273.41 sequences/s | -81.44% |
| Classifier initialisation | 88.152 ms | 29.064 ms | -67.03% |
| Model size | 2.310 MiB | 0.813 MiB | -64.79% |

INT8 reduced the model size by `1.496 MiB`, corresponding to a `64.79%` reduction and approximately `2.84×` compression.

However, this storage benefit did not translate into faster inference on the development-machine CPU:

- INT8 mean latency was approximately `5.37×` slower than FP32.
- INT8 median latency was approximately `5.65×` slower.
- INT8 p95 latency was approximately `4.39×` slower.
- FP32 delivered approximately `5.39×` the throughput of INT8.
- INT8 classifier initialisation was faster, taking `29.064 ms` compared with `88.152 ms` for FP32.

The same pattern was observed across every validation record:

| **Record** | **FP32 mean latency** | **INT8 mean latency** | **INT8 slowdown** |
|---|---:|---:|---:|
| 114 | 0.700 | 3.667 | 5.24× slower |
| 122 | 0.674 | 3.662 | 5.44× slower |
| 209 | 0.671 | 3.670 | 5.47× slower |
| 210 | 0.680 | 3.658 | 5.38× slower |
| 231 | 0.687 | 3.648 | 5.31× slower |
| 233 | 0.685 | 3.640 | 5.31× slower |

The repeated and counterbalanced benchmark therefore showed that the slower INT8 inference was consistent across records and was not simply caused by one unusually slow timing pass.

### Saved Outputs

```text
artifacts/results/deployment_evaluation/onnx_benchmarking/
    fp32_vs_int8_benchmark.json

artifacts/figures/onnx_benchmarking/
    fp32_vs_int8_latency.png
    fp32_vs_int8_throughput.png
    fp32_vs_int8_model_size.png
    per_record_mean_latency.png
```

### Key Lesson

Quantisation does not automatically make a model faster.

Dynamic INT8 quantisation reduced the ONNX model from `2.310 MiB` to `0.813 MiB`, a `64.79%` reduction, while the previous evaluations showed that its prediction behaviour and classification quality remained highly comparable to FP32.

However, the controlled inference benchmark showed that INT8 was slower on the development-machine x86 CPU. Mean latency increased from `0.681 ms` to `3.658 ms`, while throughput fell from `1,473.46` to `273.41` sequences per second.

This shows why model size, predictive quality and runtime performance must be evaluated separately. A quantised model can provide a storage advantage without providing a computational-speed advantage on a particular hardware and runtime combination.

Section 4 now provides the complete deployment evidence for both model precisions: 
FP32 baseline performance, INT8 model-size reduction, inference agreement, ground-truth classification impact and controlled computational-performance comparison. The next stage is to evaluate both deployment candidates on the target Raspberry Pi hardware.

## Milestone 29 — Raspberry Pi Target-Hardware Streaming Runtime Validation

### Implemented

- Added `validate_edge_streaming_runtime.py` to validate the production streaming-inference path on the physical Raspberry Pi 5.
- Ran both FP32 and INT8 ONNX models through the full streaming pipeline using record `114` and 36-sample chunks.
- Added lightweight event accumulation and integrity checks for:
  - valid class labels and indices;
  - finite `(4,)` logits;
  - strictly increasing target R-peaks;
  - correct flush behaviour.
- Confirmed FP32 and INT8 traversed the same target beats without treating class disagreements as runtime failures.
- Added Raspberry Pi hardware-health monitoring for:
  - available RAM;
  - CPU temperature;
  - throttling status.
- Recorded the target runtime environment, including ARM64 architecture, Python version, ONNX Runtime version and execution provider.
- Added unit tests for event integrity, flush handling, FP32-vs-INT8 comparison, telemetry parsing and overall PASS/FAIL behaviour.

### Results

The complete record `114` runtime validation passed on the Raspberry Pi.

| **Metric** | **Result** |
|---|---:|
| Samples processed | 650,000 |
| Chunks processed | 18,056 |
| FP32 prediction events | 1,873 |
| INT8 prediction events | 1,873 |
| Target peaks identical | Yes |
| FP32-vs-INT8 class agreement | 98.99% |
| FP32 integrity failures | 0 |
| INT8 integrity failures | 0 |
| Final validation status | **PASSED** |

Hardware health remained acceptable throughout the run:

| **Metric** | **Before** | **After** |
|---|---:|---:|
| Available RAM | 664.64 MiB | 656.30 MiB |
| CPU temperature | 47.2 °C | 63.1 °C |
| Throttled | `0x0` | `0x0` |

No throttling condition was observed.

### Saved Output

```text
artifacts/results/deployment_evaluation/edge_runtime_validation/
    record_114_edge_runtime_validation.json
```

### Key Lesson

The existing streaming-inference pipeline runs successfully on the Raspberry Pi with both FP32 and INT8 ONNX models.

Both precisions processed the full ECG record, emitted predictions for the same target beats, passed all event-integrity checks and completed without observed throttling.

Milestone 29 confirms target-hardware compatibility. The next step is to benchmark FP32 and INT8 inference directly on the Raspberry Pi.

## Milestone 30 — Raspberry Pi FP32 vs INT8 Inference Performance Benchmark

### Implemented

- Added `benchmark_edge_quantized_inference.py` to benchmark FP32 and dynamically quantised INT8 ONNX inference directly on the Raspberry Pi 5.
- Reused the controlled timing methodology from the earlier FP32-vs-INT8 benchmark, including:
  - identical streaming-emitted `BeatSequence` inputs;
  - 100 warm-up calls;
  - five repeated benchmark passes;
  - counterbalanced FP32/INT8 execution order.
- Benchmarked all six validation records: `114`, `122`, `209`, `210`, `231`, `233`.
- Compared both models over the same `14,556` sequences.
- Processed one record at a time to reduce Raspberry Pi memory usage while reusing the same FP32 and INT8 classifier sessions across records and repeats.
- Measured:
  - mean, median and p95 model-stage inference latency;
  - throughput;
  - classifier initialisation time;
  - model size;
  - per-record latency;
  - run-to-run variation.
- Added a direct comparison with the earlier development-machine benchmark to check whether the FP32-vs-INT8 ordering changed on ARM hardware.
- Recorded Raspberry Pi health context before and after benchmarking, including available RAM, CPU temperature, throttling state, CPU governor and CPU frequency.
- Generated FP32-vs-INT8 latency, throughput, model-size and per-record latency figures.

### Results

The Raspberry Pi benchmark reproduced the same overall behaviour seen on the development machine: INT8 remained substantially smaller, but FP32 was considerably faster at steady-state inference.

| **Metric** | **FP32** | **INT8** | **INT8 vs FP32** |
|:---:|:---:|:---:|:---:|
| Mean latency | **1.332 ms** | 3.850 ms | +189.09% |
| Median latency | **1.323 ms** | 3.849 ms | +190.95% |
| p95 latency | **1.367 ms** | 3.863 ms | +182.55% |
| Throughput | **750.884 seq/s** | 259.737 seq/s | -65.41% |
| Classifier initialisation | 76.913 ms | **38.823 ms** | INT8 faster |
| Model size | 2.310 MiB | **0.813 MiB** | -64.79% |

INT8 therefore retained the `64.79%` model-size reduction (`2.84×` compression), but its mean inference latency was approximately `2.89×` that of FP32 on the Raspberry Pi.

The same latency ordering was observed across every validation record:

| **Record** | **FP32 mean latency** | **INT8 mean latency** |
|:---:|:---:|:---:|
| 114 | **1.334 ms** | 3.847 ms |
| 122 | **1.331 ms** | 3.851 ms |
| 209 | **1.327 ms** | 3.849 ms |
| 210 | **1.332 ms** | 3.850 ms |
| 231 | **1.337 ms** | 3.852 ms |
| 233 | **1.333 ms** | 3.852 ms |

INT8 did initialise faster, taking `38.823 ms` compared with `76.913 ms` for FP32, but this is a one-time start-up cost rather than the continuous per-sequence inference cost.

The Raspberry Pi remained unthrottled before and after the benchmark. The recorded CPU governor was `ondemand`, the health snapshots reported a 2.4 GHz CPU frequency, and temperature increased from `51.0 °C` to `69.2 °C`.

### Saved Outputs

```text
artifacts/results/deployment_evaluation/edge_onnx_benchmarking/
    raspberry_pi_fp32_vs_int8_benchmark.json
    raspberry_pi_fp32_vs_int8_benchmark_raw.npz
    record_114_fp32_vs_int8_benchmark.json
    record_114_fp32_vs_int8_benchmark_raw.npz
```

Saved figures:

```text
artifacts/figures/edge_onnx_benchmarking/
    fp32_vs_int8_latency.png
    fp32_vs_int8_throughput.png
    fp32_vs_int8_model_size.png
    per_record_mean_latency.png
```

### Key Lesson

The storage benefit of dynamic INT8 quantisation did not translate into faster inference on the target Raspberry Pi hardware.

FP32 achieved substantially lower mean, median and p95 model-stage latency and approximately `2.89×` the throughput of INT8 across the complete validation benchmark. INT8 remained valuable as the smaller model and also initialised faster, but FP32 was the stronger candidate for continuous model-stage inference.

This target-hardware result confirmed that the slower INT8 behaviour observed on the development machine was not reversed on the Raspberry Pi. The next step is to test whether these model-stage differences translate into meaningful real-time behaviour when the complete streaming pipeline is paced at the ECG signal's true arrival rate.

## Milestone 31 — Raspberry Pi Real-Time Streaming and CPU Governor Evaluation

### Implemented

- Added `benchmark_edge_realtime_streaming.py` to benchmark the complete production streaming pipeline on the Raspberry Pi at the ECG signal's real arrival rate.
- Replayed record `114` using:
  - 360 Hz sampling;
  - 36-sample chunks;
  - a 100 ms chunk period;
  - 18,056 paced chunks across the full 650,000-sample record.
- Anchored each chunk to an absolute schedule from the replay start so processing delays could not shift later nominal arrival times.
- Recorded per-chunk:
  - scheduled arrival time;
  - actual processing start time;
  - processing completion time;
  - processing latency.
- Added real-time timing statistics for:
  - mean, median, p95 and maximum chunk-processing latency;
  - scheduling lateness;
  - deadline utilisation;
  - deadline misses and deadline lateness.
- Timed final stream flushing separately from normal chunk processing.
- Added Raspberry Pi health snapshots around each benchmark run.
- Added `edge_realtime_streaming_plots.py` for latency distributions, scheduling-lateness plots and direct CPU-governor comparison.
- Added unit tests for the paced scheduling logic, deadline calculations, timing boundaries, flush accounting and runtime-light imports.
- Repeated the FP32 and INT8 paced benchmark under both the `ondemand` and `performance` CPU governors.

### Results

Each run processed the complete record using a 100 ms real-time chunk deadline.

| **Metric** | **FP32 — ondemand** | **FP32 — performance** | **INT8 — ondemand** | **INT8 — performance** |
|:---:|:---:|:---:|:---:|:---:|
| Mean chunk latency | 1.7506 ms | **1.3732 ms** | 2.0410 ms | **1.6349 ms** |
| Maximum chunk latency | 112.6148 ms | **78.4367 ms** | 135.5550 ms | **92.9398 ms** |
| Maximum scheduling lateness | 12.7748 ms | **0.0724 ms** | 35.7326 ms | **0.0742 ms** |
| Deadline misses | 164 | **0** | 178 | **0** |
| Deadline miss rate | 0.9083% | **0.0000%** | 0.9858% | **0.0000%** |
| Worst deadline lateness | +12.6722 ms | **-21.5089 ms** | +35.6119 ms | **-7.0065 ms** |

A negative worst deadline-lateness value means every chunk completed before its deadline. Under the `performance` governor, FP32 retained at least 21.51 ms of deadline headroom and INT8 retained at least 7.01 ms.

The `ondemand` governor produced repeated latency excursions beyond the 100 ms chunk period, causing 164 FP32 and 178 INT8 deadline misses. Switching to `performance` removed all deadline misses for both models and reduced maximum scheduling lateness to below 0.08 ms.

All four runs completed with valid prediction-event integrity and no observed throttling. The `performance` runs also completed at the full 2.4 GHz governor configuration.

### Saved Outputs

```text
artifacts/results/deployment_evaluation/edge_realtime_streaming/
    record_114_fp32_paced.json
    record_114_int8_paced.json

artifacts/results/deployment_evaluation/edge_realtime_streaming_perf_governor/
    record_114_fp32_paced.json
    record_114_int8_paced.json
```

Saved figures:

```text
artifacts/figures/edge_realtime_streaming/
    governor_comparison_latency.png
    ondemand_paced_latency_distribution.png
    ondemand_paced_scheduling_lateness.png
    performance_paced_latency_distribution.png
    performance_paced_scheduling_lateness.png
```

### Key Lesson

A pipeline can be fast on average while still failing real-time requirements because rare latency spikes can exceed the next chunk's deadline.

The original `ondemand` configuration produced isolated deadline misses for both FP32 and INT8 even though typical chunk-processing latency was far below the 100 ms budget. Running the Raspberry Pi with the `performance` governor removed these misses, greatly reduced scheduling jitter and kept the worst processing latency below the chunk period for both precisions.

The `performance` governor was therefore selected for the remaining sustained Raspberry Pi deployment evaluation.

## Milestone 32 — Raspberry Pi Sustained Resource and Stability Evaluation

### Implemented

- Added `monitor_edge_sustained_resources.py` to evaluate the production streaming pipeline over sustained real-time Raspberry Pi runs.
- Reused the existing real-time pacing logic so ECG chunks continued to arrive according to their true 360 Hz signal timing.
- Cycled through the validation records: `114`, `122`, `209`, `210`, `231`, `233`.
- Ran matched 60-minute sustained evaluations for FP32 and INT8 ONNX using the `performance` CPU governor.
- Added periodic Raspberry Pi telemetry collection for:
  - CPU temperature and frequency;
  - process and system CPU utilisation;
  - process RSS and available RAM;
  - throttling state.
- Continued recording streaming behaviour including:
  - processed chunks and predictions;
  - processing latency and scheduling lateness;
  - deadline misses and minimum deadline margin;
  - prediction-event integrity.
- Added RSS trend estimation using total RSS change, fitted MiB/hour trend and residual scatter.
- Added correlation of deadline misses with the nearest hardware-telemetry sample for later investigation.
- Added `edge_sustained_resource_plots.py` for sustained telemetry and FP32-vs-INT8 comparisons.
- Added an extended 210-minute FP32 endurance run to investigate longer-term memory behaviour.
- Added unit tests for the sustained monitoring and plotting utilities.

### Results

Both matched 60-minute runs processed 36,000 real-time chunks without a deadline miss or prediction-integrity failure.

| Metric | FP32 — 60 min | INT8 — 60 min |
|:---:|:---:|:---:|
| Chunks processed | 36,000 | 36,000 |
| Predictions | 4,329 | 4,329 |
| Deadline misses | **0** | **0** |
| Integrity failures | **0** | **0** |
| Maximum processing latency | **84.59 ms** | 97.57 ms |
| Minimum deadline margin | **15.36 ms** | 2.38 ms |
| Mean process CPU | **3.46%** | 4.67% |
| Mean system CPU | **0.88%** | 1.18% |
| Maximum temperature | 49.6 °C | 49.6 °C |
| Mean temperature | 47.41 °C | 47.40 °C |
| RSS change | +6.70 MiB | +7.63 MiB |
| Fitted RSS trend | +5.42 MiB/hour | +4.85 MiB/hour |
| Throttling observed | No | No |

The two runs produced an identical 4,329 predictions because R-peak detection occurs before model inference. XQRS therefore processed the same signal and emitted the same beats for both precisions, ensuring that both models classified the same sequences.

FP32 retained substantially more worst-case real-time headroom. Its slowest chunk required 84.59 ms compared with 97.57 ms for INT8, leaving minimum deadline margins of 15.36 ms and 2.38 ms respectively. Both met the 100 ms deadline throughout the hour, but INT8 operated much closer to the limit.

FP32 also used less process CPU on average: 3.46% of one logical core compared with 4.67% for INT8. Whole-system CPU utilisation remained low for both runs, supporting the conclusion that background system load was limited during the experiment.

Thermal behaviour was effectively identical. Both runs averaged approximately 47.4 °C, reached 49.6 °C and remained at the full 2.4 GHz CPU frequency without observed throttling.

Process RSS increased by 6.70 MiB for FP32 and 7.63 MiB for INT8, with fitted trends of +5.42 and +4.85 MiB/hour. These trends are reported descriptively rather than being treated as evidence of a memory leak.

### Extended FP32 Endurance Run

FP32 was subsequently streamed for 210 minutes to investigate whether the shorter-run memory trend continued.

| Metric | FP32 — 210 min |
|:---:|:---:|
| Paced duration | 210 minutes |
| Chunks processed | 126,002 |
| Predictions | 16,381 |
| Deadline misses | **0** |
| Integrity failures | **0** |
| Maximum processing latency | 79.97 ms |
| Minimum deadline margin | 19.98 ms |
| Mean process CPU | 3.52% |
| Mean system CPU | 0.89% |
| Mean temperature | 47.91 °C |
| Maximum temperature | 50.7 °C |
| RSS change | +8.80 MiB |
| Fitted RSS trend | +1.41 MiB/hour |
| Throttling observed | No |

The extended run processed more than 126,000 paced chunks without a deadline miss, integrity failure or throttling event.

RSS reached 255.55 MiB after 210 minutes, an increase of 8.80 MiB. However, the fitted trend fell to approximately +1.41 MiB/hour and the telemetry shows the initial increase progressively flattening.

The longer run therefore provides stronger evidence that the one-hour RSS slope should not be extrapolated indefinitely. It does not prove the absence of a memory leak, but no runaway memory growth was observed over the tested 3.5-hour period.

### Saved Outputs

```text
artifacts/results/deployment_evaluation/edge_sustained_resources/
    fp32_sustained_60min.json
    fp32_sustained_60min_raw.npz
    int8_sustained_60min.json
    int8_sustained_60min_raw.npz
    fp32_sustained_210min.json
    fp32_sustained_210min_raw.npz
```

Saved figures:

```text
artifacts/figures/edge_sustained_resources/
    fp32_sustained_timeseries_60min.png
    int8_sustained_timeseries.png
    fp32_sustained_timeseries_210min.png
    sustained_fp32_vs_int8_comparison.png
```

### Key Lesson

Passing a short real-time benchmark is not enough to show that an edge deployment remains healthy over time.

Both FP32 and INT8 sustained one hour of real-time ECG streaming with zero deadline misses, zero integrity failures, stable temperatures and no throttling. FP32 nevertheless retained substantially more deadline headroom and required less CPU, reinforcing the earlier finding that dynamic INT8 quantisation does not improve computational performance on this Raspberry Pi.

The 210-minute FP32 run strengthened the stability evidence further: more than 126,000 chunks were processed without a missed deadline or throttling event, while the initially rising RSS progressively flattened and its fitted trend fell to approximately 1.41 MiB/hour.

Overall, the production streaming pipeline remained stable during sustained Raspberry Pi operation under the `performance` governor and produced the thermal, CPU, memory and real-time evidence needed for the final FP32-vs-INT8 deployment decision.
