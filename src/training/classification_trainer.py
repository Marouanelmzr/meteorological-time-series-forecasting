from pathlib import Path

from hydra.utils import instantiate

from src.evaluation.classification_metrics import ClassificationMetrics
from src.visualization.classification_plots import ClassificationPlots


class ClassificationTrainer:

    def __init__(self, cfg):

        self.cfg = cfg

        self.dataset = None
        self.model = None

        self.X_train = None
        self.y_train = None

        self.X_val = None
        self.y_val = None
        self.metadata = None

        self.y_pred = None
        self.y_prob = None

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
        )

    def _validate(self):

        print("Running validation...")

        self.y_pred = self.model.predict(
            self.X_val
        )

        if hasattr(self.model, "predict_proba"):

            self.y_prob = self.model.predict_proba(
                self.X_val
            )[:, 1]

        self.metrics = ClassificationMetrics(
            y_true=self.y_val,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
        )

        self.plots = ClassificationPlots(
            y_true=self.y_val,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
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

        print("Saving figures...")

        self.plots.save_all()

        if "lead_time" in self.metadata.columns:

            self.plots.performance_by_lead_time(
                self.metadata["lead_time"]
            )

        if "icao" in self.metadata.columns:

            self.plots.performance_by_station(
                self.metadata["icao"]
            )

        if (
            "lead_time" in self.metadata.columns
            and "icao" in self.metadata.columns
        ):

            self.plots.leadtime_station_heatmap(
                self.metadata["lead_time"],
                self.metadata["icao"],
            )

        print("Saving model...")

        model_path = (
            Path(self.cfg.paths.models_dir)
            / "model.joblib"
        )

        self.model.save(model_path)

        print()

        print("=" * 60)
        print("Validation results")
        print("=" * 60)

        print(self.metrics)

        print("=" * 60)