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
    arome_gust: np.ndarray

    def __post_init__(self):
        self.y_true = np.asarray(self.y_true, dtype=float)
        self.y_pred = np.asarray(self.y_pred, dtype=float)
        self.arome_gust = np.asarray(self.arome_gust, dtype=float)

        self.metrics = {}

    def compute(self):

        errors = self.y_pred - self.y_true

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

        self.metrics["bias"] = np.mean(errors)

        self.metrics["error_std"] = np.std(errors)

        self.metrics["max_error"] = np.max(np.abs(errors))

        observed_gust = self.arome_gust + self.y_true
        corrected_gust = self.arome_gust + self.y_pred

        self.metrics["raw_arome_mae"] = mean_absolute_error(
            observed_gust,
            self.arome_gust,
        )

        self.metrics["corrected_mae"] = mean_absolute_error(
            observed_gust,
            corrected_gust,
        )

        self.metrics["raw_arome_rmse"] = np.sqrt(
            mean_squared_error(
                observed_gust,
                self.arome_gust,
            )
        )

        self.metrics["corrected_rmse"] = np.sqrt(
            mean_squared_error(
                observed_gust,
                corrected_gust,
            )
        )

        self.metrics["raw_arome_r2"] = r2_score(
            observed_gust,
            self.arome_gust,
        )

        self.metrics["corrected_r2"] = r2_score(
            observed_gust,
            corrected_gust,
        )

        raw_mae = self.metrics["raw_arome_mae"]
        raw_rmse = self.metrics["raw_arome_rmse"]
        
        self.metrics["mae_improvement_percent"] = (
            100 * (raw_mae - self.metrics["corrected_mae"]) / raw_mae
            if raw_mae > 0
            else 0.0
        )
        
        self.metrics["rmse_improvement_percent"] = (
            100 * (raw_rmse - self.metrics["corrected_rmse"]) / raw_rmse
            if raw_rmse > 0
            else 0.0
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

        if not self.metrics:
            self.compute()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)