from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.evaluation.classification_metrics import ClassificationMetrics
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:  # wandb is optional at import time
    wandb = None

try:
    import optuna
except ImportError:  # optuna is optional at import time
    optuna = None


LOWER_IS_BETTER = {"val_loss", "far", "brier", "log_loss"}


class BaseTorchClassifier(BaseModel):

    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 50,
        patience: int = 10,
        monitor_metric: str = "pr_auc",
        probability_threshold: float = 0.5,
        pos_weight: Optional[float] = None,  # None -> auto: negatives / positives
        use_weighted_sampler: bool = False,
        scheduler: Optional[str] = None,  # "plateau" | "cosine" | "onecycle" | None
        grad_clip_norm: Optional[float] = 1.0,
        use_amp: Optional[bool] = None,  # None -> auto (True on cuda)
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        random_state: int = 42,
        log_to_wandb: bool = True,
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            monitor_metric=monitor_metric,
            probability_threshold=probability_threshold,
            pos_weight=pos_weight,
            use_weighted_sampler=use_weighted_sampler,
            scheduler=scheduler,
            grad_clip_norm=grad_clip_norm,
            use_amp=use_amp,
            weight_decay=weight_decay,
            random_state=random_state,
            **kwargs,
        )
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.probability_threshold = probability_threshold
        self.pos_weight = pos_weight
        self.use_weighted_sampler = use_weighted_sampler
        self.scheduler_name = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.log_to_wandb = log_to_wandb and wandb is not None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = (self.device == "cuda") if use_amp is None else use_amp
        self.scaler = StandardScaler()

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.feature_names: Optional[List[str]] = None
        self.history: List[Dict[str, float]] = []

    # Architecture hook
    def build_model(self, input_dim: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement build_model(input_dim).")

    # Data helpers
    def _to_numpy(self, X, fit: bool = False):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X.astype(np.float32)

    def _remember_feature_names(self, X) -> None:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

    # Scheduler
    def _make_scheduler(self, optimizer, steps_per_epoch: int):
        """Returns (scheduler, step_per_batch)."""
        if self.scheduler_name is None:
            return None, False
        if self.scheduler_name == "plateau":
            return (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=max(1, self.patience // 2)
                ),
                False,
            )
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs), False
        if self.scheduler_name == "onecycle":
            return (
                torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.lr, steps_per_epoch=steps_per_epoch, epochs=self.epochs
                ),
                True,
            )
        raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    # Class imbalance
    def _compute_pos_weight(self, y_train: np.ndarray) -> float:
        if self.pos_weight is not None:
            return self.pos_weight
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        if positives == 0:
            logger.warning("No positive samples in y_train; falling back to pos_weight=1.0")
            return 1.0
        return negatives / positives

    def _make_train_loader(self, X_train: np.ndarray, y_train: np.ndarray) -> DataLoader:
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
        if not self.use_weighted_sampler:
            return DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=(self.device == "cuda"),
                num_workers=4,
                persistent_workers=True,
            )

        positives = y_train.sum()
        negatives = len(y_train) - positives
        w_pos = 1.0 / max(positives, 1)
        w_neg = 1.0 / max(negatives, 1)
        sample_weights = np.where(y_train == 1, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(train_ds, batch_size=self.batch_size, sampler=sampler, pin_memory=(self.device == "cuda"), num_workers=4, persistent_workers=True,)

    # Metrics
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
        preds = (probs >= threshold).astype(int)
        return ClassificationMetrics(y_true=y_true, y_pred=preds, y_prob=probs).compute()

    def _monitor_value(self, val_loss: float, val_metrics: Dict[str, float]) -> float:
        """Lower-is-better scalar, so early stopping / pruning logic stays
        uniform regardless of which metric is being tracked."""
        if self.monitor_metric == "val_loss":
            return val_loss
        value = float(val_metrics[self.monitor_metric])
        return value if self.monitor_metric in LOWER_IS_BETTER else -value

    # Training
    def fit(self, X_train, y_train, X_val=None, y_val=None, trial: Optional["optuna.trial.Trial"] = None):
        self._remember_feature_names(X_train)
        X_train_np = self._to_numpy(X_train, fit=True)
        y_train_np = np.asarray(y_train, dtype=np.float32)

        self.input_dim = X_train_np.shape[1]
        self.model = self.build_model(self.input_dim).to(self.device)

        train_loader = self._make_train_loader(X_train_np, y_train_np)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        pos_weight_value = self._compute_pos_weight(y_train_np)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=self.device))

        scheduler, step_per_batch = self._make_scheduler(optimizer, len(train_loader))
        scaler = torch.amp.GradScaler(enabled=self.use_amp)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_np = self._to_numpy(X_val)
            y_val_np = np.asarray(y_val, dtype=np.float32)
            
            val_ds = TensorDataset(
                torch.from_numpy(X_val_np),
                torch.from_numpy(y_val_np).unsqueeze(1),
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=(self.device == "cuda"),
            )
        elif trial is not None:
            logger.warning("trial was passed to fit() but no validation data was given; pruning is disabled.")

        best_state = None
        best_monitor = float("inf")
        wait = 0
        self.history = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss, n_batches = 0.0, 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=False,
            )

            for xb, yb in pbar:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                    logits = self.model(xb, None)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()

                if self.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None and step_per_batch:
                    scheduler.step()

                running_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(
                    loss=f"{running_loss / n_batches:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

            train_loss = running_loss / max(n_batches, 1)
            epoch_log: Dict[str, float] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }

            if has_val:
                self.model.eval()

                val_losses = []
                val_probs_list = []

                with torch.no_grad():
                    for xb, yb in tqdm(
                        val_loader,
                        desc="Validation",
                        leave=False,
                    ):
                    
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)

                        with torch.amp.autocast(
                            device_type=self.device,
                            enabled=self.use_amp,
                        ):
                            logits = self.model(xb, None)
                            loss = criterion(logits, yb)

                        val_losses.append(loss.item())
                        val_probs_list.append(torch.sigmoid(logits).cpu())

                val_loss = float(np.mean(val_losses))
                val_probs = torch.cat(val_probs_list).numpy().ravel()

                val_metrics = self._compute_metrics(y_val_np, val_probs, self.probability_threshold)
                epoch_log["val_loss"] = val_loss
                # only scalar rates/scores go to the epoch log/W&B — tn/fp/fn/tp
                # stay in val_metrics for monitor_metric lookups but would just
                # clutter per-epoch charts.
                epoch_log.update({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)})

                if scheduler is not None and not step_per_batch:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                monitor_value = self._monitor_value(val_loss, val_metrics)
                if monitor_value < best_monitor:
                    best_monitor = monitor_value
                    best_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1

                if trial is not None:
                    if optuna is None:
                        raise ImportError("optuna is not installed but a trial was passed to fit().")
                    trial.report(-monitor_value, step=epoch)  # optuna: higher-is-better
                    if trial.should_prune():
                        logger.info("Trial pruned at epoch %d (%s=%.4f)", epoch, self.monitor_metric, monitor_value)
                        raise optuna.TrialPruned()

            elif scheduler is not None and not step_per_batch:
                scheduler.step()

            self.history.append(epoch_log)

            if self.log_to_wandb and wandb.run is not None:
                wandb.log(epoch_log)

            if has_val:
                logger.info(
                    "epoch %d | train_loss=%.4f | val_loss=%.4f | val_%s=%.4f",
                    epoch, train_loss, epoch_log["val_loss"], self.monitor_metric, val_metrics[self.monitor_metric],
                )
            else:
                logger.info("epoch %d | train_loss=%.4f", epoch, train_loss)

            if has_val and wait >= self.patience:
                logger.info("Early stopping at epoch %d (best %s=%.4f)", epoch, self.monitor_metric, best_monitor)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    # Inference
    def predict_logits(self, X):

        X_np = self._to_numpy(X)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_np)),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
            num_workers=4,
            persistent_workers=True,
        )

        outputs = []

        self.model.eval()

        with torch.no_grad():
            for (xb,) in loader:

                xb = xb.to(self.device, non_blocking=True)

                logits = self.model(xb, None)

                outputs.append(logits.cpu())

        return torch.cat(outputs).numpy().ravel()

    def predict_proba(self, X) -> np.ndarray:
        probs = 1.0 / (1.0 + np.exp(-self.predict_logits(X)))
        return np.column_stack([1 - probs, probs])

    def predict(self, X, threshold: Optional[float] = None) -> np.ndarray:
        threshold = self.probability_threshold if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # ClassificationPlots integration
    def evals_result(self) -> Dict[str, Dict[str, List[float]]]:
        train_loss = [h["train_loss"] for h in self.history]
        val_loss = [h["val_loss"] for h in self.history if "val_loss" in h]
        return {"validation_0": {"logloss": train_loss}, "validation_1": {"logloss": val_loss}}

    # Persistence — overrides BaseModel's joblib-based save/load, since a
    # torch module needs its state_dict plus architecture metadata
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "feature_names": self.feature_names,
                "params": self.get_params(),
                "scaler": self.scaler,
            },
            path,
        )

    def load(self, path) -> "BaseTorchClassifier":
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.feature_names = checkpoint.get("feature_names")
        self.scaler = checkpoint["scaler"]
        self.model = self.build_model(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        return self

