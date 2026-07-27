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

