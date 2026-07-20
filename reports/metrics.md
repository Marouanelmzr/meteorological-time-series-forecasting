# Classification Metrics

## True Positive (TP)

A positive sample that is correctly predicted as positive.

Example:

- Actual: Gust
- Predicted: Gust

---

## True Negative (TN)

A negative sample that is correctly predicted as negative.

Example:

- Actual: No gust
- Predicted: No gust

---

## False Positive (FP)

A negative sample that is incorrectly predicted as positive.

Also called a **false alarm**.

Example:

- Actual: No gust
- Predicted: Gust

---

## False Negative (FN)

A positive sample that is incorrectly predicted as negative.

Also called a **missed detection**.

Example:

- Actual: Gust
- Predicted: No gust

---

## Accuracy

The proportion of all predictions that are correct.

Formula:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Range: **0 to 1**

Higher values are better.

---

## Balanced Accuracy

The average of the recall for each class.

Formula:

```text
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

Range: **0 to 1**

Higher values are better.

---

## Precision

The proportion of predicted positive samples that are actually positive.

Formula:

```text
Precision = TP / (TP + FP)
```

Range: **0 to 1**

Higher values are better.

---

## Recall (Sensitivity / POD)

The proportion of actual positive samples that are correctly detected.

Formula:

```text
Recall = TP / (TP + FN)
```

Range: **0 to 1**

Higher values are better.

---

## F1 Score

The harmonic mean of Precision and Recall.

Formula:

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

Range: **0 to 1**

Higher values are better.

---

## Matthews Correlation Coefficient (MCC)

A correlation coefficient between predicted and true labels that considers TP, TN, FP and FN.

Formula:

```text
(TP × TN − FP × FN)
-----------------------------------------------
√((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

Range:

- **+1** : Perfect prediction
- **0** : Random prediction
- **−1** : Total disagreement

---

## Probability of Detection (POD)

Another name for Recall.

Formula:

```text
POD = TP / (TP + FN)
```

Range: **0 to 1**

Higher values are better.

---

## False Alarm Ratio (FAR)

The proportion of predicted positive events that are false alarms.

Formula:

```text
FAR = FP / (TP + FP)
```

Range: **0 to 1**

Lower values are better.

---

## Critical Success Index (CSI)

Measures the fraction of observed and/or predicted positive events that were correctly predicted.

Formula:

```text
CSI = TP / (TP + FP + FN)
```

Range: **0 to 1**

Higher values are better.

---

## True Skill Statistic (TSS)

Measures the model's ability to separate positive and negative samples.

Formula:

```text
TSS = Recall − False Positive Rate
```

or equivalently

```text
TSS = TP / (TP + FN) − FP / (FP + TN)
```

Range:

- **+1** : Perfect prediction
- **0** : No skill
- **−1** : Worse than random

---

## Heidke Skill Score (HSS)

Measures the improvement over random predictions.

Formula:

```text
HSS = 2 × (TP × TN − FP × FN)
      --------------------------------------------
      (TP+FN)(FN+TN) + (TP+FP)(FP+TN)
```

Range:

- **1** : Perfect prediction
- **0** : No skill
- **< 0** : Worse than random

---

## ROC AUC

Area Under the Receiver Operating Characteristic Curve.

Measures the model's ability to rank positive samples higher than negative samples across all thresholds.

Range:

- **1.0** : Perfect
- **0.5** : Random

Higher values are better.

---

## PR AUC

Area Under the Precision-Recall Curve.

Measures the trade-off between Precision and Recall across all thresholds.

Range:

- **1.0** : Perfect
- **0.0** : Worst

Higher values are better.

---

## Brier Score

Measures the mean squared error between predicted probabilities and the true labels.

Formula:

```text
Brier Score = mean((predicted_probability − true_label)²)
```

Range:

- **0** : Perfect probability estimates

Lower values are better.

---

## Log Loss

Measures the quality of predicted probabilities by penalizing incorrect and overconfident predictions.

Formula:

```text
Log Loss = −mean(y × log(p) + (1 − y) × log(1 − p))
```

Range:

- **0** : Perfect prediction

Lower values are better.