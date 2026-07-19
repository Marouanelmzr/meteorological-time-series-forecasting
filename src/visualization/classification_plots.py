from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.metrics import f1_score, precision_score, recall_score

from src.visualization.base_plots import BasePlots


class ClassificationPlots(BasePlots):

    def __init__(
        self,
        y_true,
        y_pred,
        y_prob=None,
        metadata=None,
        model=None,
        feature_names=None,
        save_dir="outputs",
    ):

        super().__init__(save_dir)

        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_prob = np.asarray(y_prob) if y_prob is not None else None

        self.metadata = metadata

        self.model = model
        self.feature_names = feature_names

    def _metadata_column(self, key):
        """Return metadata[key] as a numpy array, or None if unavailable."""

        if self.metadata is None:
            return None

        try:
            values = self.metadata[key]
        except (KeyError, TypeError):
            return None

        return np.asarray(values)

    def confusion_matrix(self):

        fig, ax = plt.subplots(figsize=(6, 6))

        ConfusionMatrixDisplay.from_predictions(
            self.y_true,
            self.y_pred,
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )

        ax.set_title("Confusion Matrix")

        self.save("confusion_matrix.png")


    def confusion_matrix_normalized(self):

        fig, ax = plt.subplots(figsize=(6, 6))

        ConfusionMatrixDisplay.from_predictions(
            self.y_true,
            self.y_pred,
            cmap="Blues",
            colorbar=False,
            normalize="true",
            values_format=".2%",
            ax=ax,
        )

        ax.set_title("Confusion Matrix (Row-Normalized)")

        self.save("confusion_matrix_normalized.png")


    def roc_curve(self):

        if self.y_prob is None:
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        RocCurveDisplay.from_predictions(
            self.y_true,
            self.y_prob,
            ax=ax,
        )

        ax.set_title("ROC Curve")

        self.save("roc_curve.png")


    def precision_recall_curve(self):

        if self.y_prob is None:
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        PrecisionRecallDisplay.from_predictions(
            self.y_true,
            self.y_prob,
            ax=ax,
        )

        ax.set_title("Precision Recall Curve")

        self.save("precision_recall_curve.png")


    def calibration_curve(self):

        if self.y_prob is None:
            return

        fig, ax = plt.subplots(figsize=(6, 6))

        CalibrationDisplay.from_predictions(
            self.y_true,
            self.y_prob,
            n_bins=10,
            ax=ax,
        )

        ax.set_title("Calibration Curve")

        self.save("calibration_curve.png")


    def probability_histogram(self):

        if self.y_prob is None:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            self.y_prob,
            bins=40,
        )

        ax.set_xlabel("Predicted probability")

        ax.set_ylabel("Count")

        ax.set_title("Predicted Probability Distribution")

        self.save("probability_histogram.png")


    def feature_importance(self):

        if self.model is None:
            return

        if hasattr(self.model, "feature_importances_"):

            importance = self.model.feature_importances_

        elif hasattr(self.model, "coef_"):

            importance = np.abs(self.model.coef_)

            if importance.ndim > 1:
                importance = importance[0]

        else:
            return

        names = (
            self.feature_names
            if self.feature_names is not None
            else [f"Feature {i}" for i in range(len(importance))]
        )

        order = np.argsort(importance)

        fig, ax = plt.subplots(figsize=(8, 10))

        ax.barh(
            np.array(names)[order],
            importance[order],
        )

        ax.set_title("Feature Importance")

        self.save("feature_importance.png")


    def threshold_analysis(self, thresholds=None):

        if self.y_prob is None:
            return

        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 19)

        precisions, recalls, f1s = [], [], []

        for t in thresholds:

            preds_at_t = (self.y_prob >= t).astype(int)

            precisions.append(
                precision_score(self.y_true, preds_at_t, zero_division=0)
            )

            recalls.append(
                recall_score(self.y_true, preds_at_t, zero_division=0)
            )

            f1s.append(
                f1_score(self.y_true, preds_at_t, zero_division=0)
            )

        best_idx = int(np.argmax(f1s))

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(thresholds, precisions, marker="o", label="Precision")
        ax.plot(thresholds, recalls, marker="o", label="Recall")
        ax.plot(thresholds, f1s, marker="o", label="F1")

        ax.axvline(
            thresholds[best_idx],
            color="gray",
            linestyle="--",
            label=f"Best F1 @ {thresholds[best_idx]:.2f}",
        )

        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold Analysis")
        ax.legend()

        self.save("threshold_analysis.png")


    def performance_by_lead_time(self):

        lead_times = self._metadata_column("lead_time")

        if lead_times is None:
            return

        scores = []

        for lead in sorted(np.unique(lead_times)):

            mask = lead_times == lead

            scores.append(
                f1_score(
                    self.y_true[mask],
                    self.y_pred[mask],
                    zero_division=0,
                )
            )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(sorted(np.unique(lead_times)), scores, marker="o")

        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("F1 Score")
        ax.set_title("Performance by Lead Time")

        self.save("performance_by_lead_time.png")


    def performance_by_station(self):

        stations = self._metadata_column("icao")

        if stations is None:
            return

        unique_stations = np.unique(stations)

        scores = []
        counts = []

        for station in unique_stations:

            mask = stations == station

            scores.append(
                f1_score(
                    self.y_true[mask],
                    self.y_pred[mask],
                    zero_division=0,
                )
            )

            counts.append(int(mask.sum()))

        order = np.argsort(scores)

        fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(unique_stations))))

        bars = ax.barh(
            unique_stations[order],
            np.array(scores)[order],
        )

        for bar, n in zip(bars, np.array(counts)[order]):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" n={n:,}",
                va="center",
                fontsize=8,
            )

        ax.set_xlabel("F1 Score")
        ax.set_title("Performance by Station")

        self.save("performance_by_station.png")


    def leadtime_station_heatmap(self):

        lead_times = self._metadata_column("lead_time")
        stations = self._metadata_column("icao")

        if lead_times is None or stations is None:
            return

        unique_leads = sorted(np.unique(lead_times))
        unique_stations = sorted(np.unique(stations))

        matrix = np.full((len(unique_stations), len(unique_leads)), np.nan)

        for i, station in enumerate(unique_stations):

            for j, lead in enumerate(unique_leads):

                mask = (stations == station) & (lead_times == lead)

                if mask.sum() == 0:
                    continue

                matrix[i, j] = f1_score(
                    self.y_true[mask],
                    self.y_pred[mask],
                    zero_division=0,
                )

        masked = np.ma.masked_invalid(matrix)

        fig, ax = plt.subplots(
            figsize=(max(8, 0.4 * len(unique_leads)), max(6, 0.3 * len(unique_stations)))
        )

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="lightgray")

        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(range(len(unique_leads)))
        ax.set_xticklabels(unique_leads, rotation=90)

        ax.set_yticks(range(len(unique_stations)))
        ax.set_yticklabels(unique_stations)

        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("Station")
        ax.set_title("F1 Score by Lead Time and Station")

        fig.colorbar(im, ax=ax, label="F1 Score")

        self.save("leadtime_station_heatmap.png")


    def save_all(self):

        self.confusion_matrix()

        self.confusion_matrix_normalized()

        self.roc_curve()

        self.precision_recall_curve()

        self.calibration_curve()

        self.probability_histogram()

        self.feature_importance()

        self.threshold_analysis()

        if self.metadata is not None:

            self.performance_by_station()

            self.performance_by_lead_time()

            self.leadtime_station_heatmap()