# Wind Gust Forecasting Using Machine Learning and Probabilistic Modeling

# 1. Introduction

Accurate forecasting of wind gusts is a challenging problem in numerical weather prediction (NWP). Although modern atmospheric models such as AROME provide direct gust diagnostics, these forecasts often exhibit systematic biases and struggle to represent local effects caused by terrain, coastal influences, and small-scale atmospheric processes.

The objective of this project is therefore **not to replace the AROME model**, but to investigate whether machine learning can be used as a post-processing technique to improve gust forecasts using historical observations and additional meteorological information.

The project progressively evolved through several stages. Initially, wind gust prediction was formulated as a binary classification problem (gust/no gust). After extensive experimentation, feature engineering, and model optimization, it became clear that the main limitation was not necessarily the choice of machine learning algorithm but rather the formulation of the prediction problem itself. This eventually motivated a transition toward probabilistic and distributional regression models capable of jointly modeling gust occurrence and gust intensity.

This report summarizes the complete experimental process, the lessons learned from each stage, and the rationale behind the current research direction.

# 2. Dataset Preparation and Data Quality Investigation

Before developing any machine learning model, considerable effort was devoted to understanding and validating the meteorological dataset. Several issues discovered during this stage had a much larger impact than changing machine learning algorithms.

## 2.1 Data Sources

The dataset combines multiple information sources:

* **AROME numerical weather prediction forecasts**
* **NOAA archived METAR observations**
* **Station metadata** (latitude, longitude, elevation)
* Historical METAR observations used to compute rolling statistics

Each sample corresponds to an AROME forecast valid at a specific station and forecast lead time.

The primary meteorological variables include:

* 10 m wind speed
* 10 m wind direction
* 10 m U/V wind components
* 850 hPa and 950 hPa wind fields
* Temperature
* Relative humidity
* Surface pressure
* Planetary Boundary Layer Height (PBLH)
* Turbulent Kinetic Energy (TKE)
* Eddy Dissipation Rate (EDR)
* AROME 60-minute gust diagnostic

## 2.2 NOAA–METAR Gust Reporting Investigation

One of the most important discoveries of the project occurred during dataset validation.

According to the METAR reporting standard, wind gusts are typically reported only when they exceed approximately **10 m/s**. However, the archived NOAA observations contained numerous gust values below this threshold.

Initially, this discrepancy raised concerns about the consistency of the target variable. As a first preprocessing experiment, all gust observations below **10 m/s** were reclassified as non-events in order to match the operational METAR reporting convention. This reduced the proportion of positive samples from approximately **3.06%** to **1.13%**, creating an extremely imbalanced binary classification problem.

Although this preprocessing was consistent with the METAR standard, it also discarded information contained in weaker gust events. Since the long-term objective of the project evolved from binary classification toward **distributional regression**, this thresholding strategy was ultimately abandoned.

The final dataset therefore retains all observed gust magnitudes, including values below **10 m/s**. Rather than treating these observations as noise, they are considered informative samples from the lower tail of the gust distribution. This allows probabilistic models to learn the complete conditional distribution of gust intensity instead of an artificially truncated one.

Operational decision thresholds (e.g., 10 m/s for issuing a gust warning) can then be applied **after prediction** during post-processing, rather than being hard-coded into the training target.

This evolution reflects an important shift in the project philosophy: instead of forcing the data to fit a binary classification framework, the objective became to preserve as much physical information as possible and allow the statistical model to learn the full distribution of wind gusts.

## 2.3 Lead Time Correction

Another important preprocessing issue was discovered during exploratory analysis.

The forecast lead time had originally been computed as

```python
lead_time = valid_time.hour
```

This incorrectly assigned midnight forecasts a lead time of **0 hours** instead of **24 hours**, causing approximately **4.2%** of the dataset to contain erroneous lead times.

The lead time computation was corrected before subsequent experiments, ensuring that the temporal relationship between forecasts and observations was physically consistent.

## 2.4 Chronological Dataset Split

To avoid temporal information leakage, all experiments used a chronological split:

| Dataset    | Period      |
| ---------- | ----------- |
| Training   | Before 2024 |
| Validation | 2024        |
| Test       | 2025        |

This split better reflects an operational forecasting scenario where models are trained on historical years and evaluated on unseen future observations.

# 3. Baseline: Raw AROME Gust Forecast

Before introducing machine learning, the raw AROME gust diagnostic (`arome_gust60_speed`) was evaluated as a reference forecast.

The gust diagnostic was thresholded at **10 m/s**, matching the METAR reporting criterion.

| Metric            |     Value |
| ----------------- | --------: |
| Accuracy          |     0.841 |
| Balanced Accuracy | **0.742** |
| Recall (POD)      | **0.641** |
| FAR               |     0.947 |
| TSS               | **0.485** |
| HSS               |     0.075 |

**Full run metrics (W&B):**

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.841 | | MCC | 0.151 |
| Balanced Accuracy | 0.742 | | CSI | 0.051 |
| Precision | 0.053 | | FN | 942 |
| Recall (POD) | 0.641 | | FP | 30,216 |
| F1 | 0.097 | | TN | 163,608 |
| TSS | 0.485 | | TP | 1,679 |
| HSS | 0.075 | | Threshold | 10 m/s |
| FAR | 0.947 | | ROC-AUC / PR-AUC | not applicable (deterministic threshold, no probability score) |

## Interpretation

Although this baseline produces an extremely large number of false alarms (FAR ≈ 95%), it successfully detects many actual gust events.

This baseline is important because it represents the operational forecast already available to meteorologists.

The objective of machine learning is therefore **not simply to achieve higher accuracy**, but to improve discrimination while reducing false alarms and preserving the ability to detect rare gust events.


# 4. Binary Classification Experiments

The first stage of the project formulated gust prediction as a binary classification problem.

Target:

```
has_gust
```

All models used the same feature family and were evaluated on the same chronological split.

## 4.1 Logistic Regression

Logistic Regression served as the simplest baseline.

Despite its simplicity, it provided surprisingly competitive probabilistic predictions.

After threshold tuning:

| Metric            | Value |
| ----------------- | ----: |
| Balanced Accuracy | 0.600 |
| ROC-AUC           | 0.877 |
| PR-AUC            | 0.150 |
| MCC               | 0.214 |

**Full run metrics (W&B, threshold = 0.85, L2 penalty, C = 1, class_weight = balanced):**

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.969 | | HSS | 0.206 |
| Balanced Accuracy | 0.653 | | CSI | 0.123 |
| Precision | 0.165 | | FAR | 0.835 |
| Recall | 0.328 | | FN | 1,761 |
| F1 | 0.220 | | FP | 4,345 |
| ROC-AUC | 0.857 | | TN | 189,479 |
| PR-AUC | 0.143 | | TP | 860 |
| MCC | 0.218 | | TSS | 0.306 |

### Interpretation

The strong ROC-AUC suggests that a significant fraction of the predictive signal is approximately linear.

This made Logistic Regression an excellent sanity-check baseline throughout the project.

## 4.2 Random Forest

Random Forest achieved extremely high overall accuracy.

However,

| Metric   |    Value |
| -------- | -------: |
| Accuracy |    98.6% |
| Recall   | **3.3%** |

**Full run metrics (W&B, 300 trees, class_weight = balanced, threshold = 0.5):**

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.986 | | HSS | 0.059 |
| Balanced Accuracy | 0.516 | | CSI | 0.032 |
| Precision | 0.401 | | FAR | 0.599 |
| Recall | 0.033 | | FN | 2,534 |
| F1 | 0.061 | | FP | 130 |
| ROC-AUC | 0.877 | | TN | 193,694 |
| PR-AUC | 0.150 | | TP | 87 |
| MCC | 0.112 | | TSS | 0.033 |

### Interpretation

At first glance, Random Forest appears to perform exceptionally well.

In reality, the model simply predicts **"no gust"** for almost every sample.

Because gust events represent only around 3% of the dataset, predicting the majority class almost everywhere minimizes classification error.

This experiment clearly demonstrates why **accuracy is an inappropriate metric for rare-event forecasting**.

## 4.3 XGBoost

XGBoost became the main tree-based model investigated throughout the project.

Several experiments were conducted, including:

* default parameters
* manual tuning
* Optuna hyperparameter optimization
* threshold optimization
* feature engineering
* station-specific training

One representative model achieved:

| Metric            | Value |
| ----------------- | ----: |
| Balanced Accuracy | 0.633 |
| ROC-AUC           | 0.892 |
| PR-AUC            | 0.188 |
| Recall            | 0.689 |

**Full run metrics (W&B, closest logged run: 800 est., depth 6, lr 0.05, threshold = 0.55, `splits_rolling_features`):**

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.977 | | HSS | 0.267 |
| Balanced Accuracy | 0.633 | | CSI | 0.140 |
| Precision | 0.219 | | FAR | 0.781 |
| Recall | 0.280 | | FN | 1,886 |
| F1 | 0.246 | | FP | 2,621 |
| ROC-AUC | 0.875 | | TN | 191,203 |
| PR-AUC | 0.165 | | TP | 735 |
| MCC | 0.236 | | TSS | 0.267 |

These experiments demonstrated that XGBoost is considerably better suited to highly imbalanced meteorological datasets than Random Forest.


# 5. Hyperparameter Optimization

A substantial amount of work was devoted to hyperparameter optimization using **Optuna**.

The search explored parameters including:

* learning rate
* number of estimators
* tree depth
* subsampling
* column sampling
* minimum child weight
* gamma
* L1 regularization
* L2 regularization

The best trial achieved a validation **PR-AUC of approximately 0.212**.

## Interpretation

Although Optuna consistently improved model performance, the improvements remained relatively modest.

This led to an important conclusion:

> Improving the **representation of the data** was considerably more beneficial than spending large amounts of time optimizing hyperparameters.

This observation motivated subsequent work on feature engineering rather than continued parameter tuning.

# 6. Feature Engineering

Rather than changing algorithms, several experiments focused on improving the information available to the models.

## 6.1 Rolling METAR Features

Meteorological variables exhibit strong temporal persistence.

To capture recent atmospheric evolution, rolling statistics were computed from previous METAR observations, including:

* rolling mean
* rolling maximum
* rolling standard deviation
* temporal trend

over 3-hour and 6-hour windows.

These features were designed to provide short-term temporal context unavailable from a single forecast timestep.

Although they improved discrimination slightly, they did not fundamentally solve the rare-event prediction problem.

**Full run metrics (W&B, XGBoost on `splits_rolling_features`, threshold sweep):**

| Threshold | Accuracy | Bal. Accuracy | F1 | Precision | Recall | ROC-AUC | PR-AUC | MCC | TSS | HSS | CSI | FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.977 | 0.633 | 0.246 | 0.219 | 0.280 | 0.875 | 0.165 | 0.236 | 0.267 | 0.267 | 0.140 | 0.781 |
| 0.65 | 0.976 | 0.637 | 0.246 | 0.215 | 0.288 | 0.888 | 0.171 | 0.237 | 0.274 | 0.274 | 0.140 | 0.785 |

## 6.2 Additional Wind Representation

Experiments also evaluated whether including explicit U and V wind components alongside speed and direction would improve performance.

The observed improvements were negligible.

This suggests that gradient-boosted trees were already able to extract most of the directional information from the existing wind representation.

**Full run metrics (W&B, XGBoost, threshold = 0.65, `splits_rolling_features`):**

| Feature set | Accuracy | Bal. Accuracy | F1 | Precision | Recall | ROC-AUC | PR-AUC | MCC | TSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Speed + direction only | 0.976 | 0.637 | 0.246 | 0.215 | 0.288 | 0.888 | 0.171 | 0.237 | 0.274 |
| Speed + direction + explicit U/V | 0.977 | 0.629 | 0.242 | 0.219 | 0.271 | 0.886 | 0.164 | 0.232 | 0.257 |

# 7. Station-Specific Experiments

One hypothesis was that a single national model could not adequately capture the diverse meteorological environments across Morocco.

To investigate this, separate XGBoost models were trained for different station groups.

Performance varied substantially between regions.

Balanced accuracy ranged from approximately:

**0.54 to 0.70**

depending on the station cluster.

**Full run metrics (W&B, XGBoost per station cluster):**

| Station cluster | Threshold | Accuracy | Bal. Accuracy | F1 | Precision | Recall | ROC-AUC | PR-AUC | MCC | TSS | HSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gmag_gmad_gmat_gmmx | 0.25 | 0.983 | 0.663 | 0.230 | 0.174 | 0.339 | 0.869 | 0.122 | 0.235 | 0.327 | 0.222 |
| gmff_gmfm_gmfo_gmmc | 0.05 | 0.985 | 0.586 | 0.177 | 0.174 | 0.180 | 0.833 | 0.088 | 0.169 | 0.172 | 0.169 |
| gmfi_gmfb_gmfk_gmmz | 0.05 | 0.950 | 0.609 | 0.166 | 0.123 | 0.254 | 0.799 | 0.098 | 0.154 | 0.218 | 0.144 |
| geml_gmmw_gmta | 0.05 | 0.985 | 0.544 | 0.079 | 0.067 | 0.096 | 0.772 | 0.037 | 0.073 | 0.088 | 0.072 |
| gmme_gmmp_gmmn_gmtt | 0.25 | 0.974 | 0.697 | 0.318 | 0.259 | 0.412 | 0.898 | 0.223 | 0.314 | 0.395 | 0.306 |
| gmme (alone) | 0.05 | 0.966 | 0.655 | 0.288 | 0.255 | 0.330 | 0.841 | 0.173 | 0.273 | 0.309 | 0.271 |
| gmmi (alone) | 0.10 | 0.876 | 0.594 | 0.218 | 0.183 | 0.271 | 0.705 | 0.174 | 0.158 | 0.189 | 0.154 |

## Interpretation

These experiments demonstrated that gust predictability is highly location dependent.

Coastal stations, inland stations, mountainous stations, and desert stations exhibit different wind regimes that are difficult for a single global model to represent equally well.

This finding suggests that future models should explicitly account for station-specific behavior rather than assuming all stations follow identical statistical relationships.

# 8. Threshold Optimization

Because the dataset is extremely imbalanced, selecting the probability threshold is almost as important as selecting the machine learning algorithm itself.

Multiple probability thresholds were evaluated for XGBoost.

**Full run metrics (W&B, XGBoost, same feature set, varying threshold):**

| Threshold | Recall | FAR | Precision | F1 | TSS | Bal. Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.280 | 0.781 | 0.219 | 0.246 | 0.267 | 0.633 |
| 0.65 | 0.288 | 0.785 | 0.215 | 0.246 | 0.274 | 0.637 |
| 0.65 (with U/V features) | 0.271 | 0.781 | 0.219 | 0.242 | 0.257 | 0.629 |

The experiments showed that:

* lower thresholds increase recall
* higher thresholds reduce false alarms
* the "best" threshold depends on the operational objective

This is particularly important in meteorology, where missing a hazardous gust event may be more costly than issuing an unnecessary warning.

Consequently, comparing models using only a single probability threshold can be misleading.

Metrics such as ROC-AUC and PR-AUC provide a more complete assessment of model quality.

# 9. Regression Experiments

After the classification experiments, the project explored predicting continuous gust magnitude instead of only gust occurrence.

The objective shifted toward **bias correction** of the AROME gust forecast.

Rather than predicting gust speed directly, models learned the error between the observation and the AROME forecast.

## 9.1 XGBoost Regressor

The regression model substantially reduced prediction errors relative to the raw AROME forecast.

Observed improvements included:

* approximately **92% reduction in MAE**
* approximately **75% reduction in RMSE**

The model also produced nearly zero systematic bias.

**Full run metrics (W&B, `reg:squarederror`, 1000 estimators, depth 6):**

| Metric | Corrected (model) | Raw AROME | Improvement |
|---|---:|---:|---:|
| MAE | 0.454 | 6.041 | 92.5% |
| RMSE | 1.742 | 7.113 | 75.5% |
| R² | 0.808 | −13.042 | — |
| Median AE | 0.071 | — | — |
| Max Error | 29.693 | — | — |
| Explained Variance | 0.809 | — | — |
| Bias | −0.055 | — | — |
| Error Std | 1.741 | — | — |

Derived binary event indicator (thresholded regression output):

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| Accuracy | 0.987 | | FN | 2,598 |
| Balanced Accuracy | 0.504 | | FP | 36 |
| Precision | 0.390 | | TN | 193,788 |
| Recall / POD | 0.009 | | TP | 23 |
| F1 | 0.017 | | TSS | 0.009 |
| HSS | 0.017 | | CSI | 0.009 |
| MCC | 0.057 | | FAR | 0.610 |

### Interpretation

These improvements demonstrate that machine learning can effectively correct systematic errors in the physical forecast.

However, thresholding the regression output into a binary classifier resulted in extremely poor event detection.

This confirms that a regression model alone cannot replace a dedicated event detector.

## 9.2 NGBoost

NGBoost was trained only on positive gust events.

Unlike ordinary regression, NGBoost predicts an entire probability distribution rather than only a single value.

Outputs include:

* predictive mean
* predictive variance
* confidence intervals
* negative log-likelihood

The model achieved approximately:

* 80% reduction in MAE
* 75% reduction in RMSE

while also producing calibrated uncertainty estimates.

Coverage analysis indicated slight undercoverage, suggesting that predictive intervals were somewhat too narrow.

**Full run metrics (W&B, lognormal distribution, 500 estimators, natural gradient, trained on `has_gust == 1` only):**

| Metric | Corrected (model) | Raw AROME | Improvement |
|---|---:|---:|---:|
| MAE | 2.108 | 10.370 | 79.7% |
| RMSE | 2.817 | 11.258 | 75.0% |
| R² | 0.587 | −0.826 | — |
| Median AE | 1.599 | — | — |
| Max Error | 19.579 | — | — |
| Explained Variance | 0.587 | — | — |
| Bias | −0.056 | — | — |
| Error Std | 2.817 | — | — |

Probabilistic calibration diagnostics:

| Metric | Value |
|---|---:|
| Predictive Mean | 10.314 |
| Predictive Std | 2.048 |
| 90% Interval Width | 6.645 |
| 90% Coverage | 81.5% |
| 95% Interval Width | 7.986 |
| 95% Coverage | 88.0% |
| Negative Log-Likelihood | 2.452 |

# 10. Rethinking the Prediction Problem

Perhaps the most important outcome of the project was not a particular machine learning model.

It was the realization that the original problem formulation was incomplete.

Initially, gust prediction was treated as either

* binary classification

or

* ordinary regression.

Neither formulation accurately reflects the statistical nature of the observations.

Most forecasts correspond to **no gust**, while the remaining observations follow a highly right-skewed continuous distribution.

Mathematically,

$$
P(Y=0)=1-\pi(x)
$$

and

$$
Y \mid Y>0 \sim f(y \mid \theta(x))
$$

In other words, the target is a **mixed random variable** composed of:

1. a discrete component describing whether a gust occurs, and
2. a continuous component describing gust intensity.

This realization naturally motivates **hurdle models** or **zero-inflated probabilistic models**, rather than increasingly complex binary classifiers.

# 11. Current Research Direction

The project is now transitioning toward **distributional regression**, where the objective is no longer to predict a single deterministic value.

Instead, the model predicts the parameters of an entire conditional probability distribution.

Candidate approaches include:

* NGBoost
* Gamma regression
* Log-normal regression
* Weibull regression
* GAMLSS
* Tweedie models
* Deep distribution regression
* Zero-inflated Gamma models
* Hurdle models
* Peaks-Over-Threshold (POT) with Generalized Pareto Distributions (GPD) for extreme gusts

This represents a significant conceptual shift from asking:

> "Will a gust occur?"

to asking:

> "What is the probability that a gust occurs, and if it does, what is its likely intensity and associated uncertainty?"

# 12. Conclusions

The project progressed through several successive stages, each providing important insights into both the data and the forecasting problem.

The principal conclusions are:

* Careful **dataset validation and preprocessing** (including correcting the NOAA–METAR inconsistency and the lead-time computation) had a larger impact than changing machine learning algorithms.
* Gust prediction is an **extremely imbalanced** learning problem, making accuracy an unreliable performance metric. Metrics such as PR-AUC, balanced accuracy, MCC, CSI, and TSS provide a more meaningful assessment.
* Classical machine learning models, particularly XGBoost, substantially improved probabilistic discrimination over simpler baselines, while Logistic Regression proved to be a strong linear benchmark.
* **Feature engineering**, including rolling METAR statistics, yielded greater benefits than extensive hyperparameter optimization, although these improvements alone were insufficient to overcome the rarity and complexity of gust events.
* Station-specific experiments revealed strong geographic variability, indicating that local meteorological conditions play a major role in gust predictability.
* Regression experiments demonstrated that machine learning can effectively correct systematic biases in AROME gust forecasts, but deterministic regression alone is not suitable for event detection.
* The most significant scientific outcome of the project is the recognition that wind gust forecasting is fundamentally a **zero-inflated probabilistic prediction problem** rather than a conventional classification or regression task.

Future work will therefore focus on probabilistic hurdle and distributional regression models capable of jointly modeling gust occurrence, gust magnitude, and predictive uncertainty in a unified framework.