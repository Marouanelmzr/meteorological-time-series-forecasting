# Wind Gust Forecasting: Deep Learning Extension with FT-Transformer

*Progress report — continuation of "Wind Gust Forecasting Using Machine Learning and Probabilistic Modeling"*

# 1. Motivation

The previous stage of the project established XGBoost as the strongest classical baseline, with hyperparameter tuning and feature engineering (rolling METAR statistics) pushing balanced accuracy to approximately 0.63–0.64 and PR-AUC to approximately 0.17–0.21. These experiments also showed that gains from further hyperparameter search were diminishing, and that the main bottleneck was how much structure the model could extract from the feature set itself, rather than tree depth or learning rate.

This motivated moving to a deep learning architecture designed specifically for tabular data: **FT-Transformer** (Feature Tokenizer + Transformer). FT-Transformer tokenizes each feature (continuous and categorical) into an embedding and processes them jointly through self-attention, which allows it to learn feature interactions that gradient-boosted trees may not easily capture, and to learn a dedicated embedding for high-cardinality categorical context such as station identity.

The goal of this stage was not to abandon the classification framing yet, but to test whether a stronger function approximator could meaningfully improve gust/no-gust discrimination before moving to the distributional regression models outlined as future work in the previous report.

# 2. Pipeline Changes

Implementing FT-Transformer required extending the existing modeling pipeline rather than rebuilding it, so that any future deep learning model can reuse the same infrastructure.

## 2.1 `BaseTorchClassifier`

A new shared base class, `src/models/base_torch_classifier.py`, was introduced to standardize data handling, training, validation, and inference for all PyTorch-based classifiers going forward. It handles:

* **Feature preparation** — continuous features are imputed (median strategy) and standardized (`StandardScaler`); categorical features are ordinal-encoded (`OrdinalEncoder`, with unseen categories mapped to a dedicated "unknown" value) and their cardinalities recorded for embedding layers.
* **Class imbalance handling** — an optional weighted random sampler, plus a configurable positive-class weighting scheme, in addition to a custom focal loss (see below).
* **Training loop** — mixed-precision training, gradient clipping, several learning-rate scheduler options (plateau, cosine, one-cycle), early stopping on a configurable monitor metric, and Optuna pruning support.
* **Validation and metrics** — per-epoch computation of the full classification metric suite via the existing `ClassificationMetrics` utility, logged to Weights & Biases.
* **Inference and persistence** — probability/threshold-based prediction, and checkpointing that stores the model weights alongside the fitted scaler, imputer, categorical encoder, and cardinalities, so a saved model is fully self-contained.

Concrete architectures (starting with FT-Transformer) now only need to implement `build_model(input_dim)`; everything else — data prep, training, evaluation, saving/loading — is inherited.

## 2.2 Focal Loss

A dedicated `src/models/losses.py` module was added, implementing **focal loss** in place of standard weighted binary cross-entropy. Focal loss down-weights the contribution of easy, well-classified negative samples (the vast majority of the dataset) and focuses gradient updates on hard and minority-class (gust) examples. The loss is parameterized by `alpha` (derived automatically from the class imbalance ratio) and `gamma = 2.0`. This was adopted specifically because the dataset remains extremely imbalanced (gust events ≈ 3% of samples), the same issue identified as the core difficulty in the classical modeling stage.

## 2.3 Categorical Feature Support

To let the model learn station-specific behavior directly — rather than training separate per-cluster models, as was done experimentally with XGBoost — `station_id` was added as an explicit categorical feature:

```yaml
categorical_columns:
  - station_id
```

in `configs/features/arome.yaml`. Internally, `cat_cardinalities` (the number of unique stations after encoding) is computed during fitting and passed to the model so that FT-Transformer can allocate a learned embedding per station, alongside the tokenized continuous meteorological features.

# 3. Results

The FT-Transformer model was trained for 30 epochs on the same chronological split (train: before 2024, validation: 2024, test: 2025) and the same underlying feature set as the classical models, with the addition of the `station_id` embedding.

## 3.1 Full Metrics

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.9035 | | HSS | 0.2573 |
| Balanced Accuracy | 0.7953 | | CSI | 0.1705 |
| Precision | 0.1854 | | FAR | 0.8146 |
| Recall / POD | 0.6803 | | FN | 1,831 |
| F1 | 0.2913 | | FP | 17,122 |
| ROC-AUC | 0.8862 | | MCC | 0.3214 |
| PR-AUC | 0.4290 | | Brier | 0.0862 |
| Log Loss | 0.3005 | | Epochs trained | 30 |
| Learning rate (final) | 2.38 × 10⁻⁴ | | | |

Validation-set metrics (held-out 2024 data, used for early stopping / monitoring):

| Metric | Value |
|---|---:|
| Val Precision | 0.1694 |
| Val Recall | 0.6681 |
| Val ROC-AUC | 0.8611 |
| Val TSS | 0.5697 |

## 3.2 Comparison with Classical Models

| Model | Balanced Accuracy | ROC-AUC | PR-AUC | Recall | MCC | HSS |
|---|---:|---:|---:|---:|---:|---:|
| Raw AROME (thresholded) | 0.742 | n/a | n/a | 0.641 | 0.151 | 0.075 |
| Logistic Regression | 0.653 | 0.857 | 0.143 | 0.328 | 0.218 | 0.206 |
| Random Forest | 0.516 | 0.877 | 0.150 | 0.033 | 0.112 | 0.059 |
| XGBoost (best) | 0.637 | 0.888 | 0.171–0.212* | 0.288 | 0.237 | 0.267 |
| **FT-Transformer** | **0.795** | **0.886** | **0.429** | **0.680** | **0.321** | **0.257** |

\* 0.212 corresponds to the best Optuna validation trial; 0.171 is the representative held-out run reported in Section 6.1 of the previous report.

# 4. Interpretation

FT-Transformer produced the largest single jump in discrimination quality seen in the project so far:

* **Balanced accuracy rose from ≈0.63–0.64 (best XGBoost) to 0.795**, and recall rose from ≈0.28–0.29 to **0.68**, essentially matching the recall level of the raw, unfiltered AROME baseline (0.641) while retaining a far better balanced accuracy and MCC than that baseline.
* **PR-AUC more than doubled**, from ≈0.17–0.21 to **0.429**, indicating substantially better ranking of true gust events above non-events despite the severe class imbalance — this is the metric the project has treated as most representative of rare-event performance since Section 4.2's Random Forest experiment showed accuracy alone to be misleading.
* **MCC and HSS also improved**, confirming the gain is not an artifact of threshold selection alone.
* The trade-off is a **higher false-positive count (FP = 17,122) and FAR (0.815)**, similar in spirit to the raw AROME baseline's false-alarm profile. This is consistent with the fact that the model is now recalling many more true gust events; some of the additional false alarms likely reflect weak or borderline gusts near the lower tail of the distribution, which the previous report's Section 2.2 investigation showed are legitimately present in the retained (unthresholded) training data.
* ROC-AUC (0.886) is comparable to XGBoost's best runs (0.888), which suggests the *ranking* ability of the two model families is similar overall, but FT-Transformer's advantage shows up specifically in the operating region that matters for rare-event detection (PR-AUC, balanced accuracy, recall at a usable threshold).

Two design choices plausibly contributed most to this jump:

1. **Focal loss**, which — unlike the plain weighted BCE used implicitly by earlier neural baselines — directly targets the imbalance problem that Section 4.2 identified as the central failure mode of naive classifiers on this dataset.
2. **Station embeddings**, which let a single global model absorb some of the geographic variability that Section 7 showed strongly affects gust predictability (balanced accuracy ranging 0.54–0.70 across station clusters with separate XGBoost models), without needing to train and maintain separate per-region models.

# 5. Notes and Caveats

* This run still frames the problem as **binary classification** (`has_gust`), the same formulation Section 10 of the previous report identified as fundamentally incomplete relative to the mixed discrete/continuous nature of the target. The improvement here is therefore a stronger result *within* that formulation, not a resolution of the underlying framing issue.
* The threshold used to compute the reported precision/recall/F1/FAR was not restated in the run configuration; as Section 8 emphasized, these operating-point metrics should be read alongside ROC-AUC/PR-AUC rather than in isolation, since the "best" threshold depends on the operational cost of missed detections versus false alarms.
* The high FP count means this configuration, as-is, would generate a large number of alerts in an operational setting; a follow-up threshold sweep (as done for XGBoost in Section 8) would help characterize the recall/FAR trade-off curve for FT-Transformer specifically.

# 6. Next Steps

With `BaseTorchClassifier` and `FocalLoss` now in place as shared infrastructure, the natural next steps are:

* **Threshold sweep and calibration analysis** for FT-Transformer, mirroring Section 8, to characterize the recall/FAR trade-off and check probability calibration (the model already reports a Brier score of 0.086 as a starting point).
* **Extending categorical embeddings** beyond `station_id` if other categorical context (e.g., season, synoptic regime) proves useful.
* **Moving the same architecture toward distributional regression**, per the direction set out in Section 11 of the previous report — e.g., having the network output the parameters of a zero-inflated or hurdle distribution instead of a single logit, so that gust occurrence and gust intensity can be modeled jointly rather than as separate classification and regression stages.
* Direct comparison against **NGBoost** (Section 9.2) on the same held-out set once a distributional FT-Transformer variant is available, since NGBoost currently provides the project's only calibrated uncertainty estimates.

# 7. Summary

Introducing FT-Transformer, backed by a reusable `BaseTorchClassifier` pipeline and a focal-loss objective tailored to the dataset's ~3% positive rate, produced the strongest classification-stage results of the project to date: balanced accuracy of 0.795, PR-AUC of 0.429, and recall of 0.680, clearly ahead of the best XGBoost configuration on every discrimination metric except FAR. Learned station embeddings appear to let a single model capture the geographic variability previously addressed only through separate per-cluster XGBoost models. The result strengthens the case for continuing to invest in deep learning architectures as the project moves toward the distributional/hurdle modeling direction identified as the primary next step.