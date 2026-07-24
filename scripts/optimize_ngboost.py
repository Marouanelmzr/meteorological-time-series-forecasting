from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import hydra
import numpy as np
import optuna

from hydra.utils import instantiate
from omegaconf import DictConfig

from sklearn.tree import DecisionTreeRegressor

from src.models.regression.ngboost import NGBoostModel


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    dataset = instantiate(cfg.dataset)

    X_train, y_train, _ = dataset.train_positive()
    X_val, y_val, _ = dataset.validation_positive()

    def objective(trial):

        params = {

            # NGBoost
            "n_estimators": trial.suggest_int(
                "n_estimators",
                500,
                3500,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.005,
                0.1,
                log=True,
            ),

            "minibatch_frac": trial.suggest_float(
                "minibatch_frac",
                0.5,
                1.0,
            ),

            "natural_gradient": True,

            "verbose": False,

            # Decision Tree base learner
            "base_max_depth": trial.suggest_int(
                "base_max_depth",
                2,
                8,
            ),

            "base_min_samples_leaf": trial.suggest_int(
                "base_min_samples_leaf",
                5,
                100,
            ),

            "base_min_samples_split": trial.suggest_int(
                "base_min_samples_split",
                2,
                50,
            ),
            
            "distribution": trial.suggest_categorical(
                "distribution",
                ["lognormal", "gamma"],
            ),

            "random_state": 42,

            "early_stopping_rounds": 50,
        }

        model = NGBoostModel(**params)

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        dist = model.predict_distribution(X_val)

        nll = -np.mean(
            dist.logpdf(y_val)
        )

        return nll

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=20,
        ),
    )

    study.optimize(
        objective,
        n_trials=100,
    )

    print("\nBest NLL:", study.best_value)

    print("\nBest parameters:\n")

    for k, v in study.best_params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()