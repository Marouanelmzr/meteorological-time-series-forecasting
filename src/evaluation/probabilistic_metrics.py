from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class ProbabilisticMetrics:

    y_true: np.ndarray
    distribution: object

    def __post_init__(self):

        self.y_true = np.asarray(self.y_true, dtype=float)

        self.metrics = {}

    def compute(self):

        # Negative Log Likelihood
        logpdf = self.distribution.logpdf(self.y_true)

        self.metrics["nll"] = float(
            -np.mean(logpdf)
        )

        # Mean prediction
        mean = self.distribution.mean()

        self.metrics["predictive_mean"] = float(
            np.mean(mean)
        )

        # Predictive standard deviation
        std = self.distribution.std()

        self.metrics["predictive_std"] = float(
            np.mean(std)
        )

        # 90% Prediction Interval
        lower90 = self.distribution.ppf(0.05)
        upper90 = self.distribution.ppf(0.95)

        inside90 = (
            (self.y_true >= lower90)
            &
            (self.y_true <= upper90)
        )

        self.metrics["coverage_90"] = float(
            inside90.mean()
        )

        self.metrics["interval_width_90"] = float(
            np.mean(
                upper90 - lower90
            )
        )

        # 95% Prediction Interval
        lower95 = self.distribution.ppf(0.025)
        upper95 = self.distribution.ppf(0.975)

        inside95 = (
            (self.y_true >= lower95)
            &
            (self.y_true <= upper95)
        )

        self.metrics["coverage_95"] = float(
            inside95.mean()
        )

        self.metrics["interval_width_95"] = float(
            np.mean(
                upper95 - lower95
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
            "Probabilistic Metrics",
            "=" * 60,
        ]

        for key, value in self.metrics.items():
            lines.append(f"{key:25s}: {value:.4f}")

        return "\n".join(lines)

    def save(self, path):

        if not self.metrics:
            self.compute()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)