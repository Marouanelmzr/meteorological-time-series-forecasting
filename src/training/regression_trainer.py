from pathlib import Path

from hydra.utils import instantiate

from src.evaluation.regression_metrics import RegressionMetrics
from src.evaluation.classification_metrics import ClassificationMetrics
from src.visualization.regression_plots import RegressionPlots


class RegressionTrainer:

    def __init__(self, cfg, logger=None):

        self.cfg = cfg
        self.logger = logger

        self.dataset = None
        self.model = None

        self.X_train = None
        self.y_train = None

        self.X_val = None
        self.y_val = None
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

        print("Loading model...")

        self.model = instantiate(self.cfg.model)

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
        )

    def _validate(self):

        print("Running validation...")

        self.y_pred = self.model.predict(
            self.X_val
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
            
            self.logger.log_metrics(metrics)
            self.logger.log_metrics(
                {f"classification/{k}": v
                 for k, v in classification_metrics.items()}
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