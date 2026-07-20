from dataclasses import dataclass

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    precision_recall_curve,
)

@dataclass
class ClassificationMetrics:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray = None

    def __post_init__(self):
        self.metrics = {}
        self.confusion_matrix = None

    def compute(self):

        tn, fp, fn, tp = confusion_matrix(
            self.y_true, self.y_pred, labels=[0, 1]
        ).ravel()

        self.confusion_matrix = np.array([[tn,fp],[fn,tp]])

        self.metrics["tn"] = int(tn)
        self.metrics["fp"] = int(fp)
        self.metrics["fn"] = int(fn)
        self.metrics["tp"] = int(tp)

        self.metrics["accuracy"] = accuracy_score(self.y_true, self.y_pred,)
        
        self.metrics["balanced_accuracy"] = balanced_accuracy_score( self.y_true, self.y_pred,)

        self.metrics["precision"] = precision_score(self.y_true, self.y_pred, zero_division=0,)

        self.metrics["recall"] = recall_score(self.y_true, self.y_pred, zero_division=0,)

        self.metrics["f1"] = f1_score(
            self.y_true,
            self.y_pred,
            zero_division=0,
        )
        
        self.metrics["mcc"] = matthews_corrcoef(self.y_true, self.y_pred,)

        self.metrics["pod"] = ( 
            tp/(tp+fn) 
            if (tp+fn) 
            else 0.0
        )

        self.metrics["far"] = (
            fp/(tp + fp )
            if(tp + fp)
            else 0.0
        )

        self.metrics["csi"] = (
            tp / (tp + fp + fn)
            if (tp + fp + fn)
            else 0.0
        )

        self.metrics["tss"] = (
            tp / (tp + fn)
            - fp / (fp + tn)
        )

        numerator = (
            2 * (tp * tn - fp * fn)
        )

        denominator = (
            (tp + fn) * (fn + tn)
            + (tp + fp) * (fp + tn)
        )

        self.metrics["hss"] = (
            numerator / denominator
            if denominator
            else 0.0
        )


        if self.y_prob is not None:

            self.metrics["roc_auc"] = roc_auc_score(
                self.y_true,
                self.y_prob,
            )

            self.metrics["pr_auc"] = average_precision_score(
                self.y_true,
                self.y_prob,
            )

            self.metrics["brier"] = brier_score_loss(
                self.y_true,
                self.y_prob,
            )

            self.metrics["log_loss"] = log_loss(
                self.y_true,
                self.y_prob,
            )

        return self.metrics
    
    def to_dict(self):
        return self.metrics
    
    def __str__(self):

        if not self.metrics:
            self.compute()

        lines = [
            "=" * 60,
            "Classification Metrics",
            "=" * 60,
        ]

        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"{key:20s}: {value:.4f}")
            else:
                lines.append(f"{key:20s}: {value}")

        return "\n".join(lines)

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)

    @property
    def cm(self):
        return self.confusion_matrix

