from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import hydra
import optuna

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.evaluation.probabilistic_metrics import ProbabilisticMetrics


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    # Load dataset once 
    dataset = instantiate(cfg.dataset)
    history_dataset = instantiate(cfg.history_dataset)

    seq_train_full, mask_train_full = history_dataset.train(dataset)
    seq_val_full, mask_val_full = history_dataset.validation(dataset)

    X_train, y_train, _ = dataset.train_positive()
    X_val, y_val, _ = dataset.validation_positive()

    train_mask = (dataset.train_df[dataset.target] > 0).to_numpy()
    val_mask = (dataset.val_df[dataset.target] > 0).to_numpy()

    seq_train = seq_train_full[train_mask]
    mask_train = mask_train_full[train_mask]
    seq_val = seq_val_full[val_mask]
    mask_val = mask_val_full[val_mask]

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

        # tabular tower (FT-Transformer)
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

        cfg.model.tab_embed_dim = trial.suggest_categorical(
            "tab_embed_dim",
            [64, 128, 192],
        )

        # sequence encoder (GRU over history)
        cfg.model.seq_hidden_dim = trial.suggest_categorical(
            "seq_hidden_dim",
            [64, 128, 192],
        )

        cfg.model.seq_num_layers = trial.suggest_int(
            "seq_num_layers",
            1,
            3,
        )

        # fusion head
        cfg.model.fusion_hidden_dim = trial.suggest_categorical(
            "fusion_hidden_dim",
            [64, 128, 256],
        )

        cfg.model.fusion_dropout = trial.suggest_float(
            "fusion_dropout",
            0.0,
            0.4,
        )

        # --- loss / output distribution ---
        # smooth_l1 excluded: it's a point-estimate loss with no predictive
        # distribution, so NLL isn't defined for it.
        cfg.model.loss_fn = trial.suggest_categorical(
            "loss_fn",
            ["gaussian_nll", "beta_nll", "student_t"],
        )

        model = instantiate(cfg.model)

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            extra_train=(seq_train, mask_train),
            extra_val=(seq_val, mask_val),
            trial=trial,
        )

        pred_dist = model.predict_distribution(
            X_val, extra_test=(seq_val, mask_val)
        )

        probabilistic_metrics = ProbabilisticMetrics(
            y_true=y_val,
            distribution=pred_dist,
        )

        nll = probabilistic_metrics.compute()["nll"]

        return nll

    study = optuna.create_study(
        direction="minimize",
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
    print("Best Validation NLL:", study.best_value)
    print("==============================\n")

    print("Best Parameters:\n")

    for k, v in study.best_params.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()