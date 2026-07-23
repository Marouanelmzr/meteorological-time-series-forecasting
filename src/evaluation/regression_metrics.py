from dataclasses import dataclass

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
)


@dataclass
class RegressionMetrics:
    y_true: np.ndarray
    y_pred: np.ndarray

    def __post_init__(self):
        self.metrics = {}

    def compute(self):

        self.metrics["mae"] = mean_absolute_error(
            self.y_true,
            self.y_pred,
        )

        self.metrics["mse"] = mean_squared_error(
            self.y_true,
            self.y_pred,
        )

        self.metrics["rmse"] = np.sqrt(
            self.metrics["mse"]
        )

        self.metrics["median_ae"] = median_absolute_error(
            self.y_true,
            self.y_pred,
        )

        self.metrics["r2"] = r2_score(
            self.y_true,
            self.y_pred,
        )

        self.metrics["explained_variance"] = explained_variance_score(
            self.y_true,
            self.y_pred,
        )

        gust_mask = self.y_true > 0

        self.metrics["gust_mae"] = mean_absolute_error(
            self.y_true[gust_mask],
            self.y_pred[gust_mask],
        )
        
        self.metrics["gust_rmse"] = np.sqrt(
            mean_squared_error(
                self.y_true[gust_mask],
                self.y_pred[gust_mask],
            )
        )

        return self.metrics

    def to_dict(self):
        return self.metrics

    def __str__(self):

        if not self.metrics:
            self.compute()

        lines = [
            "=" * 60,
            "Regression Metrics",
            "=" * 60,
        ]

        for key, value in self.metrics.items():
            lines.append(f"{key:20s}: {value:.4f}")

        return "\n".join(lines)

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)