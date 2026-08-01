from pathlib import Path

import numpy as np
from hydra.utils import instantiate

from src.evaluation.regression_metrics import RegressionMetrics
from src.evaluation.classification_metrics import ClassificationMetrics
from src.visualization.regression_plots import RegressionPlots


class RegressionTrainer:

    def __init__(self, cfg, logger=None, positive_only: bool = False):

        self.cfg = cfg
        self.logger = logger
        self.positive_only = positive_only

        self.dataset = None
        self.history_dataset = None
        self.model = None

        self.X_train = None
        self.y_train = None
        self.seq_train = None
        self.mask_train = None

        self.X_val = None
        self.y_val = None
        self.seq_val = None
        self.mask_val = None
        self.metadata = None

        self.y_pred = None

        self.metrics = None
        self.plots = None

    def train(self):

        self._prepare()

        self._fit()

        self._validate()

        self._save()

        return self.metrics

    # Internal methods

    def _prepare(self):

        print("Loading dataset...")

        self.dataset = instantiate(self.cfg.dataset)
        self.history_dataset = instantiate(self.cfg.history_dataset)

        print("Loading model...")

        self.model = instantiate(self.cfg.model)

        seq_train_full, mask_train_full = self.history_dataset.train(self.dataset)
        seq_val_full, mask_val_full = self.history_dataset.validation(self.dataset)

        if self.positive_only:
            print("Filtering to positive-target rows only...")

            train_mask = (self.dataset.train_df[self.dataset.target] > 0).to_numpy()
            val_mask = (self.dataset.val_df[self.dataset.target] > 0).to_numpy()

            (
                self.X_train,
                self.y_train,
                _,
            ) = self.dataset.train_positive()

            (
                self.X_val,
                self.y_val,
                self.metadata,
            ) = self.dataset.validation_positive()

            self.seq_train = seq_train_full[train_mask]
            self.mask_train = mask_train_full[train_mask]
            self.seq_val = seq_val_full[val_mask]
            self.mask_val = mask_val_full[val_mask]

        else:
            (
                self.X_train,
                self.y_train,
                _,
            ) = self.dataset.train()

            (
                self.X_val,
                self.y_val,
                self.metadata,
            ) = self.dataset.validation()

            self.seq_train, self.mask_train = seq_train_full, mask_train_full
            self.seq_val, self.mask_val = seq_val_full, mask_val_full

        print()

        self.dataset.summary()

        print()

    def _fit(self):

        print("Training model...")

        self.model.fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            extra_train=(self.seq_train, self.mask_train),
            extra_val=(self.seq_val, self.mask_val),
        )

    def _validate(self):

        print("Running validation...")

        self.y_pred = self.model.predict(
            self.X_val,
            extra_test=(self.seq_val, self.mask_val),
        )

        arome_gust = self.X_val["arome_gust60_speed"].values

        # To compare with classification baseline
        corrected_gust = arome_gust + self.y_pred
        observed_gust = arome_gust + self.y_val

        pred_has_gust = (corrected_gust >= 10).astype(int)
        true_has_gust = (observed_gust >= 10).astype(int)

        self.metrics = RegressionMetrics(
            y_true=self.y_val,
            y_pred=self.y_pred,
            arome_gust=arome_gust,
        )

        self.classification_metrics = ClassificationMetrics(
            y_true=true_has_gust,
            y_pred=pred_has_gust,
        )

        self.plots = RegressionPlots(
            y_true=self.y_val,
            y_pred=self.y_pred,
            metadata=self.metadata,
            model=self.model.model,
            feature_names=self.dataset.feature_names,
            save_dir=self.cfg.paths.plots_dir,
        )

    def _save(self):

        print("Saving metrics...")

        metrics_path = (
            Path(self.cfg.paths.metrics_dir)
            / "metrics.json"
        )

        self.metrics.save(metrics_path)

        classification_metrics_path = (
            Path(self.cfg.paths.metrics_dir)
            / "classification_metrics.json"
        )

        self.classification_metrics.save(classification_metrics_path)

        print("Saving figures...")

        self.plots.save_all()

        print("Saving model...")

        model_path = (
            Path(self.cfg.paths.models_dir)
            / "model.joblib"
        )

        self.model.save(model_path)

        if self.logger is not None:

            metrics = self.metrics.compute()

            classification_metrics = self.classification_metrics.compute()
            
            self.logger.log_metrics(
                {f"regression/{k}": v for k, v in metrics.items()}
            )
            self.logger.log_metrics(
                {f"classification/{k}": v for k, v in classification_metrics.items()}
            )

            self.logger.log_directory(
                self.cfg.paths.plots_dir
            )

            self.logger.log_model(
                model_path
            )

            self.logger.save_file(
                metrics_path
            )

        print()

        print("=" * 60)
        print("Validation results")
        print("=" * 60)

        print(self.metrics)

        print("=" * 60)
        print("Classification after thresholding")
        print("=" * 60)

        print(self.classification_metrics)

        print("=" * 60)