from pathlib import Path

from hydra.utils import instantiate

from src.evaluation.regression_metrics import RegressionMetrics
from src.evaluation.probabilistic_metrics import ProbabilisticMetrics

from src.visualization.regression_plots import RegressionPlots
from src.visualization.probabilistic_plots import ProbabilisticPlots


class ProbabilisticTrainer:

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
        self.pred_dist = None

        self.regression_metrics = None
        self.probabilistic_metrics = None

        self.regression_plots = None
        self.probabilistic_plots = None


    def train(self):

        self._prepare()

        self._fit()

        self._validate()

        self._save()

        return {
            "regression": self.regression_metrics,
            "probabilistic": self.probabilistic_metrics,
        }

    def _prepare(self):

        print("Loading dataset...")

        self.dataset = instantiate(self.cfg.dataset)

        print("Loading model...")

        self.model = instantiate(self.cfg.model)

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


        self.y_pred = self.model.predict(self.X_val)

        # Predictive distribution
        self.pred_dist = self.model.predict_distribution(
            self.X_val
        )

        # Regression metrics
        arome_gust = self.X_val["arome_gust60_speed"].values

        self.regression_metrics = RegressionMetrics(
            y_true=self.y_val,
            y_pred=self.y_pred,
            arome_gust=arome_gust,
        )

        # Probabilistic metrics
        self.probabilistic_metrics = ProbabilisticMetrics(
            y_true=self.y_val,
            distribution=self.pred_dist,
        )

        # Regression plots
        self.regression_plots = RegressionPlots(
            y_true=self.y_val,
            y_pred=self.y_pred,
            metadata=self.metadata,
            model=self.model.model,
            feature_names=self.dataset.feature_names,
            save_dir=self.cfg.paths.plots_dir,
        )

        # Probabilistic plots
        self.probabilistic_plots = ProbabilisticPlots(
            y_true=self.y_val,
            dist=self.pred_dist,
            metadata=self.metadata,
            model=self.model.model,
            feature_names=self.dataset.feature_names,
            X=self.X_val,
            save_dir=self.cfg.paths.plots_dir,
        )


    def _save(self):

        print("Saving metrics...")

        metrics_dir = Path(self.cfg.paths.metrics_dir)

        regression_metrics_path = (
            metrics_dir / "regression_metrics.json"
        )

        probabilistic_metrics_path = (
            metrics_dir / "probabilistic_metrics.json"
        )

        self.regression_metrics.save(
            regression_metrics_path
        )

        self.probabilistic_metrics.save(
            probabilistic_metrics_path
        )

        print("Saving figures...")

        self.regression_plots.save_all()

        self.probabilistic_plots.save_all()

        print("Saving model...")

        model_path = (
            Path(self.cfg.paths.models_dir)
            / "model.joblib"
        )

        self.model.save(model_path)

        if self.logger is not None:

            regression = self.regression_metrics.compute()

            probabilistic = self.probabilistic_metrics.compute()
            self.logger.log_metrics(
                {
                    f"regression/{k}": v
                    for k, v in regression.items()
                }
            )

            self.logger.log_metrics(
                {
                    f"probabilistic/{k}": v
                    for k, v in probabilistic.items()
                }
            )

            self.logger.log_directory(
                self.cfg.paths.plots_dir
            )

            self.logger.log_model(
                model_path
            )

            self.logger.save_file(
                regression_metrics_path
            )

            self.logger.save_file(
                probabilistic_metrics_path
            )

        print()

        print("=" * 60)
        print("Regression results")
        print("=" * 60)
        print(self.regression_metrics)

        print()

        print("=" * 60)
        print("Probabilistic results")
        print("=" * 60)
        print(self.probabilistic_metrics)

        print()

        print("=" * 60)