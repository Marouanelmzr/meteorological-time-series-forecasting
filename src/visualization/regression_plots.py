from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.visualization.base_plots import BasePlots


class RegressionPlots(BasePlots):
    def __init__(
        self,
        y_true,
        y_pred,
        metadata=None,
        model=None,
        feature_names=None,
        X=None,
        save_dir="outputs",
        target_label="Gust excess (m/s)",
        quantile_preds=None,  
        quantile_levels=None,
    ):
        super().__init__(save_dir)

        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)

        self.metadata = metadata
        self.model = model
        self.feature_names = feature_names
        self.X = X 
        self.target_label = target_label

        self.quantile_preds = (
            np.asarray(quantile_preds) if quantile_preds is not None else None
        )
        self.quantile_levels = (
            np.asarray(quantile_levels) if quantile_levels is not None else None
        )

    def _metadata_column(self, key):
        if self.metadata is None:
            return None
        try:
            values = self.metadata[key]
        except (KeyError, TypeError):
            return None
        return np.asarray(values)

    # Main

    def save_all(self):
        self.plot_prediction_scatter()
        self.plot_residuals()
        self.plot_residual_histogram()
        self.plot_target_distribution()
        self.plot_qq_residuals()
        self.plot_error_by_magnitude_bin()
        self.plot_feature_importance()

        if self.metadata is not None:
            self.plot_mae_by_station()
            self.plot_mae_by_lead_time()

        if HAS_SHAP and self.X is not None:
            self.plot_shap_summary()

        if self.quantile_preds is not None and self.quantile_levels is not None:
            self.plot_pit_histogram()
            self.plot_pinball_loss_by_level()

    # Prediction vs truth

    def plot_prediction_scatter(self):

        mn = min(self.y_true.min(), self.y_pred.min())
        mx = max(self.y_true.max(), self.y_pred.max())

        fig, ax = plt.subplots(figsize=(7, 7))

        hb = ax.hexbin(
            self.y_true, self.y_pred,
            gridsize=50, cmap="viridis", mincnt=1, bins="log",
        )
        fig.colorbar(hb, ax=ax, label="log10(count)")

        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect prediction")

        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        bias = float(np.mean(self.y_pred - self.y_true))
        corr = float(np.corrcoef(self.y_true, self.y_pred)[0, 1])

        ax.text(
            0.03, 0.97,
            f"MAE={mae:.2f}  RMSE={rmse:.2f}\nBias={bias:+.2f}  r={corr:.3f}  n={len(self.y_true):,}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(f"Observed {self.target_label}")
        ax.set_ylabel(f"Predicted {self.target_label}")
        ax.set_title("Prediction vs Observation")
        ax.legend(loc="lower right", fontsize=8)

        self.save("prediction_scatter.png")

    # Residuals

    def plot_residuals(self):

        residuals = self.y_pred - self.y_true

        fig, ax = plt.subplots(figsize=(8, 6))

        hb = ax.hexbin(
            self.y_pred, residuals,
            gridsize=50, cmap="viridis", mincnt=1, bins="log",
        )
        fig.colorbar(hb, ax=ax, label="log10(count)")

        # Binned mean residual trend (20 equal-count bins over predicted value)
        order = np.argsort(self.y_pred)
        pred_sorted = self.y_pred[order]
        resid_sorted = residuals[order]
        n_bins = 20
        bin_edges = np.array_split(np.arange(len(pred_sorted)), n_bins)
        bin_centers = [pred_sorted[idx].mean() for idx in bin_edges if len(idx) > 0]
        bin_means = [resid_sorted[idx].mean() for idx in bin_edges if len(idx) > 0]
        ax.plot(bin_centers, bin_means, color="red", linewidth=2, label="Binned mean residual")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel(f"Predicted {self.target_label}")
        ax.set_ylabel("Residual (predicted - observed)")
        ax.set_title("Residual Plot")
        ax.legend(loc="best", fontsize=8)

        self.save("residuals.png")

    def plot_residual_histogram(self):
        residuals = self.y_pred - self.y_true

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=60)
        ax.axvline(0, color="red", linestyle="--")
        ax.axvline(residuals.mean(), color="black", linestyle="-",
                   label=f"Mean = {residuals.mean():+.2f}")

        ax.set_xlabel("Residual (predicted - observed)")
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution")
        ax.legend()

        self.save("residual_histogram.png")

    def plot_qq_residuals(self):

        from scipy import stats

        residuals = self.y_pred - self.y_true
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("QQ-Plot of Residuals vs Normal")

        self.save("qq_residuals.png")

    def plot_target_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        bins = np.histogram_bin_edges(
            np.concatenate([self.y_true, self.y_pred]), bins=60
        )
        ax.hist(self.y_true, bins=bins, histtype="step", linewidth=2, label="Observed")
        ax.hist(self.y_pred, bins=bins, histtype="step", linewidth=2, label="Predicted")

        ax.legend()
        ax.set_xlabel(self.target_label)
        ax.set_ylabel("Count")
        ax.set_title("Target Distribution")

        self.save("target_distribution.png")

    # Error by magnitude — the diagnostic that matters most here

    def plot_error_by_magnitude_bin(self, n_bins: int = 8):

        edges = np.quantile(self.y_true, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)  # guard against duplicate edges at low cardinality
        bin_idx = np.digitize(self.y_true, edges[1:-1], right=True)

        maes, biases, labels, counts = [], [], [], []
        for b in range(len(edges) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            maes.append(mean_absolute_error(self.y_true[mask], self.y_pred[mask]))
            biases.append(float(np.mean(self.y_pred[mask] - self.y_true[mask])))
            labels.append(f"{edges[b]:.1f}-{edges[b+1]:.1f}")
            counts.append(int(mask.sum()))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].bar(labels, maes, color="steelblue")
        axes[0].set_ylabel("MAE")
        axes[0].set_xlabel(f"Observed {self.target_label} bin")
        axes[0].set_title("MAE by Magnitude Bin")
        axes[0].tick_params(axis="x", rotation=45)
        for i, n in enumerate(counts):
            axes[0].text(i, maes[i], f"n={n:,}", ha="center", va="bottom", fontsize=7)

        colors = ["crimson" if b < 0 else "seagreen" for b in biases]
        axes[1].bar(labels, biases, color=colors)
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_ylabel("Bias (pred - obs)")
        axes[1].set_xlabel(f"Observed {self.target_label} bin")
        axes[1].set_title("Bias by Magnitude Bin")
        axes[1].tick_params(axis="x", rotation=45)

        self.save("error_by_magnitude_bin.png")

    # Station / lead-time breakdowns

    def plot_mae_by_station(self):
        stations = self._metadata_column("icao")
        if stations is None:
            return

        unique_stations = np.unique(stations)
        maes, counts = [], []
        for s in unique_stations:
            mask = stations == s
            maes.append(mean_absolute_error(self.y_true[mask], self.y_pred[mask]))
            counts.append(int(mask.sum()))

        order = np.argsort(maes)[::-1]  # worst first

        fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(unique_stations))))
        bars = ax.barh(unique_stations[order], np.array(maes)[order])
        for bar, n in zip(bars, np.array(counts)[order]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" n={n:,}", va="center", fontsize=8)

        ax.set_xlabel("MAE")
        ax.set_title("MAE by Station")
        ax.invert_yaxis()

        self.save("mae_by_station.png")

    def plot_mae_by_lead_time(self):
        lead_times = self._metadata_column("lead_time")
        if lead_times is None:
            return

        maes = []
        leads = sorted(np.unique(lead_times))
        for lead in leads:
            mask = lead_times == lead
            maes.append(mean_absolute_error(self.y_true[mask], self.y_pred[mask]))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(leads, maes, marker="o")
        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("MAE")
        ax.set_title("MAE by Lead Time")

        self.save("mae_by_lead_time.png")

    # Feature importance

    def plot_feature_importance(self):
        if self.model is None or self.feature_names is None:
            return

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(np.ravel(self.model.coef_))
        else:
            return

        importance = pd.Series(importance, index=self.feature_names).sort_values()

        fig, ax = plt.subplots(figsize=(8, 10))
        importance.tail(20).plot.barh(ax=ax)
        ax.set_title("Top 20 Feature Importance")

        self.save("feature_importance.png")

    # SHAP — fixed: now actually receives real feature data

    def plot_shap_summary(self, max_samples: int = 2000):
        if self.X is None:
            return

        X = self.X
        if hasattr(X, "sample") and len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=0)
        elif len(X) > max_samples:
            idx = np.random.default_rng(0).choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        try:
            explainer = shap.TreeExplainer(self.model)
            values = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"SHAP explanation failed, skipping: {e}")
            return

        plt.figure(figsize=(8, 10))
        shap.summary_plot(
            values, X_sample,
            feature_names=self.feature_names, show=False, max_display=20,
        )
        self.save("shap_summary.png")

    # Quantile / distributional diagnostics (optional)

    def plot_pit_histogram(self, n_bins: int = 20):

        # Linear interpolation of the PIT value from the quantile grid
        pit_values = np.array([
            np.interp(yt, qp, self.quantile_levels)
            for yt, qp in zip(self.y_true, self.quantile_preds)
        ])
        pit_values = np.clip(pit_values, 0, 1)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(pit_values, bins=n_bins, density=True, color="steelblue", edgecolor="white")
        ax.axhline(1.0, color="red", linestyle="--", label="Uniform (calibrated)")

        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title("PIT Histogram (calibration check)")
        ax.legend()

        self.save("pit_histogram.png")

    def plot_pinball_loss_by_level(self):

        losses = []
        for i, tau in enumerate(self.quantile_levels):
            q_pred = self.quantile_preds[:, i]
            diff = self.y_true - q_pred
            loss = np.mean(np.maximum(tau * diff, (tau - 1) * diff))
            losses.append(loss)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(self.quantile_levels, losses, marker="o")
        ax.set_xlabel("Quantile level")
        ax.set_ylabel("Pinball loss")
        ax.set_title("Pinball Loss by Quantile Level")

        self.save("pinball_loss_by_level.png")