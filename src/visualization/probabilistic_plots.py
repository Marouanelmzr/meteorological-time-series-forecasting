from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.visualization.base_plots import BasePlots


class ProbabilisticPlots(BasePlots):

    def __init__(
        self,
        y_true,
        dist=None,
        quantile_preds=None,
        quantile_levels=None,
        metadata=None,
        model=None,
        feature_names=None,
        param_names=None,
        X=None,
        save_dir="outputs",
        target_label="Gust excess (m/s)",
    ):
        super().__init__(save_dir)

        self.y_true = np.asarray(y_true, dtype=float)

        self.dist = dist
        self.quantile_preds = (
            np.asarray(quantile_preds) if quantile_preds is not None else None
        )
        self.quantile_levels = (
            np.asarray(quantile_levels) if quantile_levels is not None else None
        )

        if self.dist is None and self.quantile_preds is None:
            raise ValueError(
                "ProbabilisticPlots needs either `dist` (predictive "
                "distribution object) or `quantile_preds`/`quantile_levels`."
            )

        self.metadata = metadata
        self.model = model
        self.feature_names = feature_names
        self.param_names = param_names
        self.X = X
        self.target_label = target_label

        self._mean = None
        self._std = None

    # Distributional helpers

    def _metadata_column(self, key):
        if self.metadata is None:
            return None
        try:
            values = self.metadata[key]
        except (KeyError, TypeError):
            return None
        return np.asarray(values)

    def _get_mean(self):
        if self._mean is not None:
            return self._mean
        if self.dist is not None:
            self._mean = np.asarray(self.dist.mean(), dtype=float)
        else:
            # Fall back to the median of the quantile grid as the point forecast
            mid = len(self.quantile_levels) // 2
            self._mean = self.quantile_preds[:, mid]
        return self._mean

    def _get_std(self):
        if self._std is not None:
            return self._std
        if self.dist is not None and hasattr(self.dist, "std"):
            self._std = np.asarray(self.dist.std(), dtype=float)
        elif self.quantile_preds is not None:
            # Robust std proxy from the IQR-ish spread of the quantile grid
            lo = self.quantile_preds[:, 0]
            hi = self.quantile_preds[:, -1]
            self._std = (hi - lo) / 4.0
        else:
            self._std = None
        return self._std

    def _quantile_at_level(self, tau):
        """Predicted value at quantile level `tau` for every sample."""
        if self.dist is not None:
            return np.asarray(self.dist.ppf(tau), dtype=float)

        return np.array([
            np.interp(tau, self.quantile_levels, row)
            for row in self.quantile_preds
        ])

    def _pit_values(self):
        """Probability integral transform: F(y_true) under the predictive dist."""
        if self.dist is not None and hasattr(self.dist, "cdf"):
            pit = np.asarray(self.dist.cdf(self.y_true), dtype=float)
        elif self.quantile_preds is not None:
            pit = np.array([
                np.interp(yt, qp, self.quantile_levels)
                for yt, qp in zip(self.y_true, self.quantile_preds)
            ])
        else:
            raise RuntimeError("No usable CDF or quantile grid for PIT computation.")
        return np.clip(pit, 0, 1)

    # Main

    def save_all(self):
        self.plot_prediction_scatter()
        self.plot_residuals()
        self.plot_target_distribution()
        self.plot_error_by_magnitude_bin()
        self.plot_shap_summary()

        if self.metadata is not None:
            self.plot_mae_by_station()
            self.plot_mae_by_lead_time()

        self.plot_pit_histogram()
        self.plot_calibration_reliability()
        self.plot_sharpness()
        self.plot_predicted_std_vs_abs_error()
        self.plot_pinball_loss_by_level()
        self.plot_crps_by_magnitude_bin()
        self.plot_nll_histogram()
        self.plot_example_predictive_distributions()

    # Point-forecast diagnostics (kept from RegressionPlots, driven by the mean)

    def plot_prediction_scatter(self):
        mean_pred = self._get_mean()

        mn = min(self.y_true.min(), mean_pred.min())
        mx = max(self.y_true.max(), mean_pred.max())

        fig, ax = plt.subplots(figsize=(7, 7))

        hb = ax.hexbin(
            self.y_true, mean_pred,
            gridsize=50, cmap="viridis", mincnt=1, bins="log",
        )
        fig.colorbar(hb, ax=ax, label="log10(count)")

        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect prediction")

        mae = mean_absolute_error(self.y_true, mean_pred)
        rmse = np.sqrt(mean_squared_error(self.y_true, mean_pred))
        bias = float(np.mean(mean_pred - self.y_true))
        corr = float(np.corrcoef(self.y_true, mean_pred)[0, 1])

        ax.text(
            0.03, 0.97,
            f"MAE={mae:.2f}  RMSE={rmse:.2f}\nBias={bias:+.2f}  r={corr:.3f}  n={len(self.y_true):,}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(f"Observed {self.target_label}")
        ax.set_ylabel(f"Predicted mean {self.target_label}")
        ax.set_title("Predictive Mean vs Observation")
        ax.legend(loc="lower right", fontsize=8)

        self.save("prediction_scatter.png")

    def plot_residuals(self):
        mean_pred = self._get_mean()
        residuals = mean_pred - self.y_true

        fig, ax = plt.subplots(figsize=(8, 6))

        hb = ax.hexbin(
            mean_pred, residuals,
            gridsize=50, cmap="viridis", mincnt=1, bins="log",
        )
        fig.colorbar(hb, ax=ax, label="log10(count)")

        order = np.argsort(mean_pred)
        pred_sorted = mean_pred[order]
        resid_sorted = residuals[order]
        n_bins = 20
        bin_edges = np.array_split(np.arange(len(pred_sorted)), n_bins)
        bin_centers = [pred_sorted[idx].mean() for idx in bin_edges if len(idx) > 0]
        bin_means = [resid_sorted[idx].mean() for idx in bin_edges if len(idx) > 0]
        ax.plot(bin_centers, bin_means, color="red", linewidth=2, label="Binned mean residual")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        ax.set_xlabel(f"Predicted mean {self.target_label}")
        ax.set_ylabel("Residual (mean pred - observed)")
        ax.set_title("Residual Plot (mean forecast)")
        ax.legend(loc="best", fontsize=8)

        self.save("residuals.png")

    def plot_target_distribution(self):
        mean_pred = self._get_mean()

        fig, ax = plt.subplots(figsize=(8, 6))

        bins = np.histogram_bin_edges(
            np.concatenate([self.y_true, mean_pred]), bins=60
        )
        ax.hist(self.y_true, bins=bins, histtype="step", linewidth=2, label="Observed")
        ax.hist(mean_pred, bins=bins, histtype="step", linewidth=2, label="Predicted mean")

        ax.legend()
        ax.set_xlabel(self.target_label)
        ax.set_ylabel("Count")
        ax.set_title("Target Distribution")

        self.save("target_distribution.png")

    def plot_error_by_magnitude_bin(self, n_bins: int = 8):
        mean_pred = self._get_mean()

        edges = np.quantile(self.y_true, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        bin_idx = np.digitize(self.y_true, edges[1:-1], right=True)

        maes, biases, labels, counts = [], [], [], []
        for b in range(len(edges) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            maes.append(mean_absolute_error(self.y_true[mask], mean_pred[mask]))
            biases.append(float(np.mean(mean_pred[mask] - self.y_true[mask])))
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

    def plot_mae_by_station(self):
        stations = self._metadata_column("icao")
        if stations is None:
            return

        mean_pred = self._get_mean()
        unique_stations = np.unique(stations)
        maes, counts = [], []
        for s in unique_stations:
            mask = stations == s
            maes.append(mean_absolute_error(self.y_true[mask], mean_pred[mask]))
            counts.append(int(mask.sum()))

        order = np.argsort(maes)[::-1]

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

        mean_pred = self._get_mean()
        maes = []
        leads = sorted(np.unique(lead_times))
        for lead in leads:
            mask = lead_times == lead
            maes.append(mean_absolute_error(self.y_true[mask], mean_pred[mask]))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(leads, maes, marker="o")
        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("MAE")
        ax.set_title("MAE by Lead Time")

        self.save("mae_by_lead_time.png")

    def plot_shap_summary(self, max_samples: int = 2000):

        if not HAS_SHAP:
            return
        if self.model is None or self.X is None or self.feature_names is None:
            return

        X = self.X
        if hasattr(X, "sample") and len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=0)
        elif len(X) > max_samples:
            idx = np.random.default_rng(0).choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        else:
            X_sample = X

        n_params = getattr(self.model, "n_params", None)
        if n_params is None:
            importances = getattr(self.model, "feature_importances_", None)
            if isinstance(importances, (list, tuple)):
                n_params = len(importances)
            elif isinstance(importances, np.ndarray) and importances.ndim == 2:
                n_params = importances.shape[0]
            else:
                n_params = 2  # default: most NGBoost distns (e.g. LogNormal, Gamma) have 2

        names = self.param_names or [f"param_{i}" for i in range(n_params)]

        for i, name in enumerate(names):
            try:
                explainer = shap.TreeExplainer(self.model, model_output=i)
                shap_values = explainer.shap_values(X_sample)
            except Exception as e:
                print(f"SHAP explanation failed for parameter '{name}', skipping: {e}")
                continue

            plt.figure(figsize=(8, 10))
            shap.summary_plot(
                shap_values, X_sample,
                feature_names = list(self.feature_names), show=False, max_display=20,
            )
            plt.title(f"SHAP Summary — {name}")
            self.save(f"shap_summary_{name}.png")

    # Distributional diagnostics

    def plot_pit_histogram(self, n_bins: int = 20):

        pit_values = self._pit_values()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(pit_values, bins=n_bins, density=True, color="steelblue", edgecolor="white")
        ax.axhline(1.0, color="red", linestyle="--", label="Uniform (calibrated)")

        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title("PIT Histogram (calibration check)")
        ax.legend()

        self.save("pit_histogram.png")

    def plot_calibration_reliability(self, nominal_levels=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99)):

        nominal_levels = np.asarray(sorted(nominal_levels))
        empirical = []
        for p in nominal_levels:
            lower_tau = (1 - p) / 2
            upper_tau = 1 - lower_tau
            lo = self._quantile_at_level(lower_tau)
            hi = self._quantile_at_level(upper_tau)
            covered = (self.y_true >= lo) & (self.y_true <= hi)
            empirical.append(float(np.mean(covered)))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.plot(nominal_levels, empirical, marker="o", markersize=8,
                color="steelblue", linewidth=2, label="Observed coverage")

        for p, e in zip(nominal_levels, empirical):
            ax.annotate(f"{e:.0%}", (p, e), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

        ax.set_xticks(nominal_levels)
        ax.set_xticklabels([f"{p:.0%}" for p in nominal_levels])
        ax.set_yticks(nominal_levels)
        ax.set_yticklabels([f"{p:.0%}" for p in nominal_levels])
        ax.set_xlim(nominal_levels.min() - 0.05, 1.02)
        ax.set_ylim(nominal_levels.min() - 0.05, 1.02)

        ax.set_xlabel("Nominal coverage (central prediction interval)")
        ax.set_ylabel("Empirical coverage")
        ax.set_title("Coverage Calibration Plot")
        ax.legend(loc="upper left", fontsize=8)

        self.save("calibration_reliability.png")

    def plot_sharpness(self):

        std = self._get_std()
        lo = self._quantile_at_level(0.1)
        hi = self._quantile_at_level(0.9)
        width = hi - lo

        fig, axes = plt.subplots(1, 2 if std is not None else 1, figsize=(13 if std is not None else 7, 5))
        axes = np.atleast_1d(axes)

        if std is not None:
            axes[0].hist(std, bins=60, color="steelblue")
            axes[0].set_xlabel("Predicted std")
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"Sharpness — Predicted Std (median={np.median(std):.2f})")

        ax_w = axes[-1]
        ax_w.hist(width, bins=60, color="seagreen")
        ax_w.set_xlabel("80% interval width")
        ax_w.set_ylabel("Count")
        ax_w.set_title(f"Sharpness — 80% PI Width (median={np.median(width):.2f})")

        self.save("sharpness.png")

    def plot_predicted_std_vs_abs_error(self, n_bins: int = 15):

        std = self._get_std()
        if std is None:
            return

        mean_pred = self._get_mean()
        abs_err = np.abs(mean_pred - self.y_true)

        edges = np.quantile(std, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        bin_idx = np.digitize(std, edges[1:-1], right=True)

        std_centers, mean_abs_errs = [], []
        for b in range(len(edges) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            std_centers.append(float(std[mask].mean()))
            mean_abs_errs.append(float(abs_err[mask].mean()))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(std_centers, mean_abs_errs, color="steelblue", zorder=3)
        mn = min(std_centers + mean_abs_errs)
        mx = max(std_centers + mean_abs_errs)
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="|error| = predicted std")

        ax.set_xlabel("Predicted std (binned mean)")
        ax.set_ylabel("Mean |residual|")
        ax.set_title("Predicted Std vs Actual Error Magnitude")
        ax.legend(fontsize=8)

        self.save("predicted_std_vs_abs_error.png")

    def plot_pinball_loss_by_level(self, n_levels: int = 19):
        levels = (
            self.quantile_levels if self.quantile_levels is not None
            else np.linspace(0.05, 0.95, n_levels)
        )

        losses = []
        for tau in levels:
            q_pred = self._quantile_at_level(tau)
            diff = self.y_true - q_pred
            loss = np.mean(np.maximum(tau * diff, (tau - 1) * diff))
            losses.append(loss)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(levels, losses, marker="o")
        ax.set_xlabel("Quantile level")
        ax.set_ylabel("Pinball loss")
        ax.set_title("Pinball Loss by Quantile Level")

        self.save("pinball_loss_by_level.png")

    def plot_crps_by_magnitude_bin(self, n_bins: int = 8, n_levels: int = 99):

        levels = np.linspace(0.005, 0.995, n_levels)
        pinball_per_sample = np.zeros((len(self.y_true), n_levels))
        for i, tau in enumerate(levels):
            q_pred = self._quantile_at_level(tau)
            diff = self.y_true - q_pred
            pinball_per_sample[:, i] = np.maximum(tau * diff, (tau - 1) * diff)

        crps_per_sample = 2.0 * pinball_per_sample.mean(axis=1)

        edges = np.quantile(self.y_true, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        bin_idx = np.digitize(self.y_true, edges[1:-1], right=True)

        crps_means, labels, counts = [], [], []
        for b in range(len(edges) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            crps_means.append(float(crps_per_sample[mask].mean()))
            labels.append(f"{edges[b]:.1f}-{edges[b+1]:.1f}")
            counts.append(int(mask.sum()))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, crps_means, color="steelblue")
        ax.set_ylabel("CRPS")
        ax.set_xlabel(f"Observed {self.target_label} bin")
        ax.set_title(f"CRPS by Magnitude Bin (overall mean={crps_per_sample.mean():.3f})")
        ax.tick_params(axis="x", rotation=45)
        for i, n in enumerate(counts):
            ax.text(i, crps_means[i], f"n={n:,}", ha="center", va="bottom", fontsize=7)

        self.save("crps_by_magnitude_bin.png")

    def plot_nll_histogram(self):
        """Per-sample negative log-likelihood under the fitted distribution."""
        if self.dist is None or not hasattr(self.dist, "logpdf"):
            return

        try:
            logpdf = np.asarray(self.dist.logpdf(self.y_true), dtype=float)
        except Exception as e:
            print(f"NLL computation failed, skipping: {e}")
            return

        nll = -logpdf
        nll = nll[np.isfinite(nll)]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(nll, bins=60, color="steelblue")
        ax.axvline(nll.mean(), color="red", linestyle="--",
                   label=f"Mean NLL = {nll.mean():.3f}")
        ax.set_xlabel("Negative log-likelihood")
        ax.set_ylabel("Count")
        ax.set_title("Per-sample NLL")
        ax.legend()

        self.save("nll_histogram.png")

    def plot_example_predictive_distributions(self, n_examples: int = 6, seed: int = 0):

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.y_true), size=min(n_examples, len(self.y_true)), replace=False)

        n_cols = 3
        n_rows = int(np.ceil(len(idx) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        has_pdf = self.dist is not None and hasattr(self.dist, "pdf")
        levels = np.linspace(0.01, 0.99, 200)

        for ax, i in zip(axes, idx):
            yt = self.y_true[i]

            if has_pdf:
                lo = float(self._quantile_at_level(0.001)[i])
                hi = float(self._quantile_at_level(0.999)[i])
                grid = np.linspace(lo, hi, 200)
                try:
                    density = self.dist[i].pdf(grid) if hasattr(self.dist, "__getitem__") else None
                except Exception:
                    density = None
                if density is None:
                    # Fall back to evaluating the full-batch pdf and slicing
                    try:
                        density = np.asarray(self.dist.pdf(grid[:, None]))[:, i]
                    except Exception:
                        density = None
                if density is not None:
                    ax.plot(grid, density, color="steelblue")
                    ax.fill_between(grid, density, alpha=0.2, color="steelblue")

            q_curve = np.array([self._quantile_at_level(t)[i] for t in levels])
            if not has_pdf:
                ax_twin = ax
                ax_twin.plot(q_curve, levels, color="steelblue")
                ax_twin.set_ylabel("CDF")

            ax.axvline(yt, color="red", linestyle="--", label="Observed")
            ax.set_title(f"Sample {i}")
            ax.legend(fontsize=7)

        for ax in axes[len(idx):]:
            ax.axis("off")

        fig.suptitle("Example Predictive Distributions", y=1.02)
        fig.tight_layout()

        self.save("example_predictive_distributions.png")