from pathlib import Path

from hydra.utils import instantiate

from src.evaluation.classification_metrics import ClassificationMetrics
from src.visualization.classification_plots import ClassificationPlots


class ClassificationTrainer:

    def __init__(self, cfg, logger=None,):

        self.cfg = cfg
        self.logger = logger

        self.dataset = None
        self.model = None

        self.history_dataset = None
        self.extra_train = None
        self.extra_val = None

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

        # Optional sequence/history branch — only wired up when the config
        # includes a history_dataset section (e.g. for FTTransformerWithHistory
        # runs). Absent for plain FTTransformerClassifier configs, so this is
        # a no-op there.
        history_cfg = self.cfg.get("history_dataset", None)
        if history_cfg is not None:
            print("Loading history dataset...")
            self.history_dataset = instantiate(history_cfg)
            self.extra_train = self.history_dataset.train(self.dataset)
            self.extra_val = self.history_dataset.validation(self.dataset)

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
            extra_train=self.extra_train,
            extra_val=self.extra_val,
        )

    def _validate(self):

        print("Running validation...")

        threshold = self.cfg.evaluation.probability_threshold

        if hasattr(self.model, "predict_proba"):
        
            self.y_prob = self.model.predict_proba(
                self.X_val, extra_test=self.extra_val
            )[:, 1]
        
            self.y_pred = (
                self.y_prob >= threshold
            ).astype(int)
        
        else:
        
            self.y_pred = self.model.predict(
                self.X_val, extra_test=self.extra_val
            )

        self.metrics = ClassificationMetrics(
            y_true=self.y_val,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
        )

        self.plots = ClassificationPlots(
            y_true=self.y_val,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
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

        print("Saving figures...")

        self.plots.save_all()

        print("Saving model...")

        model_path = (
            Path(self.cfg.paths.models_dir)
            / "model.joblib"
        )

        self.model.save(model_path)

        if self.logger is not None:

            self.logger.log_metrics(self.metrics.compute())
            self.logger.log_directory(self.cfg.paths.plots_dir)
            self.logger.log_model(model_path)
            self.logger.save_file(metrics_path)

        print()

        print("=" * 60)
        print("Validation results")
        print("=" * 60)

        print(self.metrics)

        print("=" * 60)