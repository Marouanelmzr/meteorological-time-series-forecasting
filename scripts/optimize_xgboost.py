from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import hydra
import optuna

from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score
from src.models.classification.xgboost import XGBoostModel


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    # Load dataset once
    dataset = instantiate(cfg.dataset)

    X_train, y_train, _ = dataset.train()
    X_val, y_val, _ = dataset.validation()

    def objective(trial):


        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),

            # Fixed parameters
            "scale_pos_weight": cfg.model.scale_pos_weight,
            "objective": cfg.model.objective,
            "eval_metric": "aucpr",
            "random_state": cfg.model.random_state,
            "n_jobs": cfg.model.n_jobs,
            "early_stopping_rounds": 50,
        }

        model = XGBoostModel(**params)

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        y_prob = model.predict_proba(X_val)[:, 1]

        return average_precision_score(
            y_val,
            y_prob,
        )

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=100,
    )

    print("\nBest PR-AUC:", study.best_value)

    print("\nBest parameters:\n")

    for k, v in study.best_params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()