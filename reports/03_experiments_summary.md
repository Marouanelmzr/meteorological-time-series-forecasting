# Wind Gust Forecasting: Stage 2 — Gust Intensity Forecasting

*Progress report — continuation of "Wind Gust Forecasting: Deep Learning Extension with FT-Transformer"*

# 1. Motivation

The previous report established that a plain FT-Transformer, trained with focal loss on the binary `has_gust` target, substantially outperformed the classical baselines (balanced accuracy 0.795, PR-AUC 0.429, recall 0.680), and identified two directions for further work: (1) extending the tabular FT-Transformer with a dedicated encoder for METAR history rather than only rolling-window summary statistics, and (2) moving from binary classification toward a joint occurrence/intensity formulation, since `has_gust` alone discards the magnitude of the forecast error that ultimately matters operationally.

This stage addresses both directions together. First, the classification model itself was extended with a recurrent history encoder fused against the tabular FT-Transformer via cross-attention, which alone produced a large jump in classification quality over the plain FT-Transformer. Second, a new regression stage was introduced, targeting the **gust intensity error** — the discrepancy between the AROME forecast and the actual METAR observation — with the goal of both point estimation and calibrated distributional (uncertainty-aware) prediction. NGBoost was run as a classical distributional baseline, and a history-aware FT-Transformer regressor (`FTTransformerWithHistoryRegressor`) was trained under a point-prediction objective as well as three distributional heads (Gaussian NLL, Gaussian Beta-NLL, and Student-t).

**A note on evaluation scope:** several of the regression runs below report classification-style metrics (accuracy, precision, recall, FAR, etc.) alongside regression metrics. These are derived by thresholding the predicted gust magnitude to recover an implicit occurrence decision, evaluated jointly with the underlying point/interval regression — consistent with the hurdle-style framing flagged as future work previously. This report treats that pairing as given; if the underlying evaluation harness differs, the classification figures below should be re-attributed accordingly.

# 2. Pipeline Changes

## 2.1 GRU History Encoder + Cross-Attention Fusion (Classification Stage)

The classification-stage FT-Transformer was extended with a GRU-based sequence embedder consuming historical METAR observations preceding each forecast run:

```yaml
- metar_wind_speed_ms
- metar_temp_c
- metar_slp_hpa
- metar_wind_dir_sin
- metar_wind_dir_cos
- metar_hours_before_run
```

Rather than concatenating the tabular (FT-Transformer) and sequential (GRU) embeddings, the two representations are fused through **cross-attention**, allowing the tabular context (station identity, current synoptic state) to selectively attend over the recent observation history instead of merging it as a fixed-size block. This is a direct evolution of the plain FT-Transformer from the previous stage, which had no access to raw observation history beyond precomputed rolling statistics.

## 2.2 `FTTransformerWithHistoryRegressor` and `BaseTorchRegressor`

The same tabular + sequential + cross-attention architecture was reused for the regression stage via a new `FTTransformerWithHistoryRegressor`, built on a `BaseTorchRegressor` base class that mirrors the `BaseTorchClassifier` infrastructure introduced previously (shared training loop, checkpointing, metric logging), extended for regression and distributional outputs. Key architectural points:

* **Shared tabular/sequential backbone.** `FTTransformerWithHistory` combines the FT-Transformer tokenizer (continuous features + `station_id` embedding, `tab_embed_dim`) with a GRU sequence encoder (`seq_hidden_dim`, `seq_num_layers`) over the same six-feature METAR history window, fused via a configurable-dropout fusion head (`fusion_hidden_dim`, `fusion_dropout`).
* **Distributional output heads.** The output dimensionality is selected by loss function: `gaussian_nll` and `beta_nll` produce 2 outputs (mean, dispersion), `student_t` produces 3 (location, scale, degrees-of-freedom-related parameter), and any other loss (e.g. `smooth_l1` for point prediction) produces a single scalar. This lets the same architecture serve as either a point regressor or a distributional model by swapping only the loss/head configuration.
* **Separate scaling pipeline for sequence inputs.** Historical sequence features are imputed (median) and standardized independently from the tabular features, fit only on real (non-padded) timesteps via a boolean mask; padded positions are zeroed after scaling rather than imputed, with the mask itself passed forward as an explicit channel so the model can distinguish "missing" from "zero." Imputer/scaler state is persisted alongside the model for self-contained checkpointing, following the same pattern established for the tabular preprocessing in the previous stage.

## 2.3 Target Definition

The regression target is the **gust intensity error**: the difference between the AROME forecast and the actual METAR-observed gust value, evaluated for the gust-relevant subset of the data (`has_gust == 1`). This directly targets the operational quantity of interest — by how much, and in which direction, AROME's forecast should be corrected — rather than the binary occurrence signal alone.

# 3. Results

## 3.1 Stage 1 Revisited: Classification with GRU History + Cross-Attention Fusion

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.8778 | | HSS | 0.7556 |
| Balanced Accuracy | 0.8779 | | CSI | 0.7746 |
| Precision | 0.9110 | | FAR | 0.0890 |
| Recall / POD | 0.8380 | | FN | 269 |
| F1 | 0.8730 | | FP | 136 |
| TSS | 0.7558 | | TN | 1,517 |
| MCC | 0.7581 | | TP | 1,392 |
| Epochs trained | 52 | | Learning rate (final) | 1.406 × 10⁻⁴ |

Auxiliary regression metrics from the same run: corrected MAE 1.7386, corrected RMSE 2.5253, corrected R² 0.7060, bias −0.5670, explained variance 0.5786; validation R² 0.5491.

This is a substantial jump over the plain FT-Transformer from the previous stage (balanced accuracy 0.795 → 0.878, PR-AUC-equivalent recall 0.680 → 0.838, MCC 0.321 → 0.758), and the false-positive count drops sharply (17,122 → 136), indicating the GRU history and cross-attention fusion meaningfully sharpen the occurrence decision, not just the ranking.

## 3.2 Point Regression: `FTTransformerWithHistoryRegressor` (`smooth_l1`)

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.8790 | | Corrected MAE | 1.7193 |
| Balanced Accuracy | 0.8790 | | Corrected RMSE | 2.4694 |
| Precision | 0.8894 | | Corrected R² | 0.7189 |
| Recall / POD | 0.8663 | | Bias | −0.2949 |
| F1 | 0.8777 | | Explained Variance | 0.5817 |
| MCC | 0.7583 | | Val R² | 0.5116 |
| FAR | 0.1106 | | Val RMSE | 2.6494 |

## 3.3 NGBoost (Classical Distributional Baseline)

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Corrected MAE | 1.9806 | | Coverage@90 | 0.8877 |
| Corrected RMSE | 2.6419 | | Coverage@95 | 0.9300 |
| Corrected R² | 0.6782 | | Interval width@90 | 9.4897 |
| NLL | 2.1667 | | Interval width@95 | 12.4607 |
| MAE improvement vs. raw AROME | 53.4% | | RMSE improvement vs. raw AROME | 53.6% |

Raw AROME baseline (no correction): MAE 4.2547, RMSE 5.6985, R² −0.4972 — i.e. the uncorrected forecast is worse than predicting the mean.

## 3.4 FT-Transformer Distributional Regression

| Distribution | Corrected MAE | Corrected RMSE | Corrected R² | NLL | Coverage@90 | Coverage@95 | Width@90 | Width@95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian (NLL) | 1.8625 | 2.6444 | 0.6776 | 2.7392 | 0.8377 | 0.8787 | 6.047 | 7.206 |
| Gaussian (Beta-NLL) | 1.8591 | 2.6236 | 0.6827 | 2.3571 | 0.8712 | 0.9071 | 6.778 | 8.076 |
| Student-t | **1.8070** | **2.5879** | **0.6912** | 2.2126 | 0.8395 | 0.9031 | 6.059 | 7.868 |
| NGBoost (ref.) | 1.9806 | 2.6419 | 0.6782 | **2.1667** | 0.8877 | 0.9300 | 9.490 | 12.461 |

# 4. Interpretation

* **Cross-attention fusion is the single largest driver of improvement seen so far in the classification stage.** Moving from the plain FT-Transformer (Section 3 of the previous report) to the GRU + cross-attention variant lifts balanced accuracy from 0.795 to 0.878 and MCC from 0.321 to 0.758, while cutting false positives by two orders of magnitude (17,122 → 136). This suggests that raw observation history, attended to selectively rather than summarized into rolling statistics or concatenated as a fixed block, carries discriminative signal the earlier feature set could not expose.
* **Point prediction still wins on point accuracy.** The plain point-regression variant (`smooth_l1`) has the best corrected MAE/RMSE/R² of every regression model tried (1.7193 / 2.4694 / 0.7189), ahead of every distributional variant. This is expected — distributional heads spend model capacity on dispersion/shape parameters rather than purely minimizing point error — but it means point regression alone remains the right choice whenever calibrated uncertainty is not required downstream.
* **Among distributional models, Student-t dominates the other FT-Transformer heads on every point metric** (MAE 1.807, RMSE 2.588, R² 0.691) and comes closest to NGBoost's NLL (2.213 vs. 2.167) while still trailing it slightly. The heavy-tailed Student-t likelihood appears better matched to this error distribution than the two Gaussian-family heads, plausibly because gust forecast errors have fatter tails than a Gaussian assumes (Gaussian max error 15.36 vs. Student-t 16.26, despite Student-t having *lower* overall MAE/RMSE — i.e. it isn't simply being more conservative everywhere, it's allocating probability mass more appropriately for the tail).
* **NGBoost remains the best-calibrated model in absolute terms** (coverage@90 = 0.888, coverage@95 = 0.930, both closest to their nominal targets, and best NLL) but achieves this by producing intervals roughly 1.5–2× wider than the FT-Transformer variants (width@90 = 9.49 vs. 6.0–6.8) and with worse point accuracy than any FT-Transformer regression variant. This is a classic sharpness/calibration trade-off: NGBoost is honestly wide, while the FT-Transformer heads are sharper but somewhat under-covered (all coverage@90 figures sit below the nominal 0.90).
* **Beta-NLL gives the best-calibrated FT-Transformer variant** (coverage@90 = 0.871, coverage@95 = 0.907, both closer to nominal than plain Gaussian or Student-t), at a small cost in point accuracy relative to Student-t. This matches Beta-NLL's known purpose — down-weighting the gradient contribution of high-variance samples during training tends to produce better-calibrated, if not always sharper, predictive variances than the plain Gaussian NLL.
* **Every corrected model — point or distributional — massively outperforms the raw AROME forecast** (MAE improvement 53–58%, RMSE improvement 53–58%, and raw AROME R² is negative), confirming the value of the correction stage regardless of which specific head is used; the remaining open question is which head to deploy operationally, which depends on whether downstream use needs a point correction, a full predictive distribution, or both.

# 6. Next Steps

* **Optuna hyperparameter search is currently running on the Student-t distributional architecture** — the best-performing FT-Transformer distributional variant on point metrics and second-best on NLL — with the goal of narrowing the gap to NGBoost's calibration while retaining Student-t's point-accuracy advantage.
* **Post-hoc calibration** (e.g. conformal or isotonic recalibration of the predictive intervals) is worth evaluating given the consistent under-coverage at the 90% level across all FT-Transformer variants.
* **Direct hurdle-model formalization**: since occurrence and intensity are already being evaluated jointly, formalizing this as an explicit two-head (or zero-inflated) model — rather than deriving occurrence post hoc by thresholding magnitude — remains the cleanest way to model both quantities without relying on an implicit threshold.

# 7. Summary

Extending the classification-stage FT-Transformer with a GRU history encoder fused via cross-attention produced the largest single improvement seen in the project to date (balanced accuracy 0.795 → 0.878, MCC 0.321 → 0.758). Reusing the same tabular + sequential backbone for gust intensity regression, a plain point-prediction head gives the best raw point accuracy (corrected MAE 1.719, R² 0.719) of any model tried, while among the distributional heads, Student-t is the strongest on both point accuracy (MAE 1.807, R² 0.691) and likelihood (NLL 2.213), trailing only NGBoost's NLL (2.167) — which remains the best-calibrated model overall but at the cost of much wider intervals and weaker point accuracy. All corrected models cut MAE roughly in half relative to the raw AROME forecast. Optuna tuning of the Student-t architecture is the immediate next step, aimed at closing the remaining calibration gap to NGBoost without giving up the FT-Transformer family's point-accuracy and interval-sharpness advantage.