from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import hydra
import optuna

from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score

from copy import deepcopy


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

        cfg.model.lr = trial.suggest_float(
            "lr",
            1e-5,
            5e-3,
            log=True,
        )

        cfg.model.weight_decay = trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True,
        )

        cfg.model.batch_size = trial.suggest_categorical(
            "batch_size",
            [512, 1024],
        )

        cfg.model.scheduler = trial.suggest_categorical(
            "scheduler",
            [
                None,
                "cosine",
                "plateau",
                "onecycle",
            ],
        )

        cfg.model.d_token = trial.suggest_categorical(
            "d_token",
            [64, 128, 192, 256],
        )

        cfg.model.n_blocks = trial.suggest_int(
            "n_blocks",
            2,
            6,
        )

        cfg.model.attention_n_heads = trial.suggest_categorical(
            "attention_n_heads",
            [4, 8],
        )

        cfg.model.ffn_d_hidden_multiplier = trial.suggest_categorical(
            "ffn_d_hidden_multiplier",
            [2.0, 4.0],
        )

        cfg.model.attention_dropout = trial.suggest_float(
            "attention_dropout",
            0.0,
            0.4,
        )

        cfg.model.ffn_dropout = trial.suggest_float(
            "ffn_dropout",
            0.0,
            0.4,
        )

        cfg.model.residual_dropout = trial.suggest_float(
            "residual_dropout",
            0.0,
            0.2,
        )

        cfg.model.focal_alpha = trial.suggest_float(
            "focal_alpha",
            0.80,
            0.99,
        )

        cfg.model.focal_gamma = trial.suggest_float(
            "focal_gamma",
            1.0,
            4.0,
        )

        model = instantiate(cfg.model)

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            trial=trial,
        )

        y_prob = model.predict_proba(X_val)[:, 1]

        return average_precision_score(
            y_val,
            y_prob,
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
        ),
    )

    study.optimize(
        objective,
        n_trials=100,
        show_progress_bar=True,
    )

    print("\n==============================")
    print("Best PR-AUC:", study.best_value)
    print("==============================\n")

    print("Best Parameters:\n")

    for k, v in study.best_params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()