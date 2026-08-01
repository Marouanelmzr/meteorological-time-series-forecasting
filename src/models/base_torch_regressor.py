from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm
from scipy.stats import norm 

from src.models.losses import BetaGaussianNLLLoss,StudentTLoss
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None

try:
    import optuna
except ImportError:
    optuna = None


LOWER_IS_BETTER = {"val_loss", "mae", "mse", "rmse"}

_LOSS_FNS = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "gaussian_nll": lambda: nn.GaussianNLLLoss(eps=1e-6),
    "beta_nll": lambda: BetaGaussianNLLLoss(beta=0.5),
    "student_t": lambda: StudentTLoss(),
}


class BaseTorchRegressor(BaseModel):

    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 50,
        patience: int = 10,
        monitor_metric: str = "mae",
        loss_fn: str = "smooth_l1",
        scheduler: Optional[str] = None,
        grad_clip_norm: Optional[float] = 1.0,
        use_amp: Optional[bool] = None,
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        random_state: int = 42,
        log_to_wandb: bool = True,
        categorical_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            monitor_metric=monitor_metric,
            loss_fn=loss_fn,
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
        self.loss_fn = loss_fn
        self.scheduler_name = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.log_to_wandb = log_to_wandb and wandb is not None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = (self.device == "cuda") if use_amp is None else use_amp
        self.scaler = StandardScaler()
        self.cat_encoder = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.imputer = SimpleImputer(strategy="median")

        self.categorical_columns = categorical_columns or []

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.feature_names: Optional[List[str]] = None
        self.history: List[Dict[str, float]] = []

    # Architecture hook
    def build_model(self, input_dim: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement build_model(input_dim).")

    # Data helpers — identical to classifier, unrelated to target type
    def _prepare_features(self, X, fit=False):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FTTransformer requires a pandas DataFrame.")

        cont_cols = [c for c in X.columns if c not in self.categorical_columns]

        X_cont = X[cont_cols]
        X_cat = X[self.categorical_columns]

        if fit:
            X_cont = self.imputer.fit_transform(X_cont)
            X_cont = self.scaler.fit_transform(X_cont)
        else:
            X_cont = self.imputer.transform(X_cont)
            X_cont = self.scaler.transform(X_cont)

        if len(self.categorical_columns) > 0:
            if fit:
                X_cat = self.cat_encoder.fit_transform(X_cat)
                self.cat_cardinalities = [len(c) for c in self.cat_encoder.categories_]
            else:
                X_cat = self.cat_encoder.transform(X_cat)
            X_cat = X_cat.astype(np.int64)
        else:
            X_cat = np.empty((len(X), 0), dtype=np.int64)
            self.cat_cardinalities = []

        return X_cont.astype(np.float32), X_cat

    def _prepare_extra_inputs(self, extra, fit: bool = False):
        return ()

    def _remember_feature_names(self, X) -> None:
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

    # Scheduler — unchanged
    def _make_scheduler(self, optimizer, steps_per_epoch: int):
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

    def _make_train_loader(self, X_train_cont, X_train_cat, y_train, extra_arrays=()):
        train_ds = TensorDataset(
            torch.from_numpy(X_train_cont),
            torch.from_numpy(X_train_cat),
            *[torch.from_numpy(a) for a in extra_arrays],
            torch.from_numpy(y_train).unsqueeze(1),
        )
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device == "cuda"),
            num_workers=4,
            persistent_workers=True,
        )

    def _make_eval_loader(self, X_cont, X_cat, y=None, extra_arrays=(), shuffle=False):
        tensors = [torch.from_numpy(X_cont), torch.from_numpy(X_cat)]
        tensors.extend(torch.from_numpy(a) for a in extra_arrays)
        if y is not None:
            tensors.append(torch.from_numpy(y).unsqueeze(1))
        ds = TensorDataset(*tensors)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=(self.device == "cuda"),
            num_workers=4,
            persistent_workers=True,
        )

    # Persistence hooks
    def _extra_state(self) -> dict:
        return {}

    def _load_extra_state(self, state: dict) -> None:
        pass

    # Metrics
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": r2_score(y_true, y_pred),
        }

    def _split_distributional(self, preds: torch.Tensor):
       mu, log_var = preds[:, 0:1], preds[:, 1:2]
       var = torch.exp(log_var.clamp(min=-10, max=10))
       return mu, var

    def _split_student_t(self, preds: torch.Tensor):
        mu = preds[:, 0:1]
        log_sigma = preds[:, 1:2]
        log_nu = preds[:, 2:3]
        sigma = F.softplus(log_sigma) + 1e-6
        nu = 2.0 + F.softplus(log_nu) 
        return mu, sigma, nu

    def _monitor_value(self, val_loss: float, val_metrics: Dict[str, float]) -> float:
        if self.monitor_metric == "val_loss":
            return val_loss
        value = float(val_metrics[self.monitor_metric])
        return value if self.monitor_metric in LOWER_IS_BETTER else -value

    # Training
    def fit(self, X_train, y_train, X_val=None, y_val=None, extra_train=None, extra_val=None, trial: Optional["optuna.trial.Trial"] = None):
        self._remember_feature_names(X_train)

        X_train_cont, X_train_cat = self._prepare_features(X_train, fit=True)
        extra_train_arrays = self._prepare_extra_inputs(extra_train, fit=True)

        y_train_np = np.asarray(y_train, dtype=np.float32)

        self.input_dim = X_train_cont.shape[1]
        self.model = self.build_model(self.input_dim).to(self.device)

        train_loader = self._make_train_loader(X_train_cont, X_train_cat, y_train_np, extra_train_arrays)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        criterion = _LOSS_FNS[self.loss_fn]()

        scheduler, step_per_batch = self._make_scheduler(optimizer, len(train_loader))
        scaler = torch.amp.GradScaler(enabled=self.use_amp)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_cont, X_val_cat = self._prepare_features(X_val, fit=False)
            extra_val_arrays = self._prepare_extra_inputs(extra_val, fit=False)
            y_val_np = np.asarray(y_val, dtype=np.float32)
            val_loader = self._make_eval_loader(
                X_val_cont, X_val_cat, y=y_val_np, extra_arrays=extra_val_arrays, shuffle=False
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

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

            for batch in pbar:
                *inputs, yb = batch
                inputs = [t.to(self.device, non_blocking=True) for t in inputs]
                yb = yb.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                    preds = self.model(*inputs)
                    if not torch.isfinite(preds).all():
                        raise RuntimeError("Invalid predictions (NaN/Inf)")
                    if self.loss_fn in ["gaussian_nll", "beta_nll"]:
                        mu, var = self._split_distributional(preds)
                        loss = criterion(mu, yb, var)
                    elif self.loss_fn == "student_t":
                        mu, sigma, nu = self._split_student_t(preds)
                        loss = criterion(mu, yb, sigma, nu)
                    else:
                        loss = criterion(preds, yb)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Loss became {loss}")

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
                pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            train_loss = running_loss / max(n_batches, 1)
            epoch_log: Dict[str, float] = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}

            if has_val:
                self.model.eval()
                val_losses = []
                val_preds_list = []

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation", leave=False):
                        *inputs, yb = batch
                        inputs = [t.to(self.device, non_blocking=True) for t in inputs]
                        yb = yb.to(self.device, non_blocking=True)

                        with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                            preds = self.model(*inputs)
                            if self.loss_fn in ("gaussian_nll", "beta_nll"):
                                mu, var = self._split_distributional(preds)
                                loss = criterion(mu, yb, var)
                                preds_for_metrics = mu
                            elif self.loss_fn == "student_t":
                                mu, sigma, nu = self._split_student_t(preds)
                                loss = criterion(mu, yb, sigma, nu)
                                preds_for_metrics = mu
                            else:
                                loss = criterion(preds, yb)
                                preds_for_metrics = preds

                        val_losses.append(loss.item())
                        val_preds_list.append(preds_for_metrics.cpu())

                val_loss = float(np.mean(val_losses))
                val_preds = torch.cat(val_preds_list).numpy().ravel()

                val_metrics = self._compute_metrics(y_val_np, val_preds)
                epoch_log["val_loss"] = val_loss
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
                    trial.report(-monitor_value, step=epoch)
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
    def predict(self, X, extra_test=None) -> np.ndarray:
        X_cont, X_cat = self._prepare_features(X, fit=False)
        extra_test_arrays = self._prepare_extra_inputs(extra_test, fit=False)

        loader = self._make_eval_loader(X_cont, X_cat, y=None, extra_arrays=extra_test_arrays, shuffle=False)

        outputs = []
        self.model.eval()

        with torch.no_grad():
            for batch in loader:
                inputs = [t.to(self.device, non_blocking=True) for t in batch]
                preds = self.model(*inputs)
                if self.loss_fn in ["gaussian_nll", "beta_nll"]:
                    preds, _ = self._split_distributional(preds)
                elif self.loss_fn == "student_t":
                    preds, _, _ = self._split_student_t(preds)
                outputs.append(preds.cpu())

        return torch.cat(outputs).numpy().ravel()

    def predict_distribution(self, X, extra_test=None):
        if self.loss_fn not in ("gaussian_nll", "beta_nll", "student_t"):
            raise ValueError(
                "predict_distribution() requires loss_fn in "
                "{'gaussian_nll','beta_nll','student_t'}."
            )

        X_cont, X_cat = self._prepare_features(X, fit=False)
        extra_test_arrays = self._prepare_extra_inputs(extra_test, fit=False)
        loader = self._make_eval_loader(X_cont, X_cat, y=None, extra_arrays=extra_test_arrays, shuffle=False)

        self.model.eval()
        if self.loss_fn == "student_t":
            from scipy.stats import t as student_t
            mus, sigmas, nus = [], [], []
            with torch.no_grad():
                for batch in loader:
                    inputs = [t.to(self.device, non_blocking=True) for t in batch]
                    preds = self.model(*inputs)
                    mu, sigma, nu = self._split_student_t(preds)
                    mus.append(mu.cpu()); sigmas.append(sigma.cpu()); nus.append(nu.cpu())
            mu = torch.cat(mus).numpy().ravel().astype(np.float64)
            sigma = torch.cat(sigmas).numpy().ravel().astype(np.float64)
            nu = torch.cat(nus).numpy().ravel().astype(np.float64)
            return student_t(df=nu, loc=mu, scale=sigma)

        # existing gaussian_nll / beta_nll branch unchanged
        mus, sigmas = [], []
        with torch.no_grad():
            for batch in loader:
                inputs = [t.to(self.device, non_blocking=True) for t in batch]
                preds = self.model(*inputs)
                mu, var = self._split_distributional(preds)
                mus.append(mu.cpu()); sigmas.append(var.sqrt().cpu())
        mu = torch.cat(mus).numpy().ravel()
        std = torch.cat(sigmas).numpy().ravel()
        return norm(loc=mu, scale=std)

    # RegressionPlots integration
    def evals_result(self) -> Dict[str, Dict[str, List[float]]]:
        train_loss = [h["train_loss"] for h in self.history]
        val_loss = [h["val_loss"] for h in self.history if "val_loss" in h]
        return {"validation_0": {"loss": train_loss}, "validation_1": {"loss": val_loss}}

    # Persistence
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
                "imputer": self.imputer,
                "cat_encoder": self.cat_encoder,
                "cat_cardinalities": self.cat_cardinalities,
                "categorical_columns": self.categorical_columns,
                "extra_state": self._extra_state(),
            },
            path,
        )

    def load(self, path) -> "BaseTorchRegressor":
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.feature_names = checkpoint.get("feature_names")
        self.scaler = checkpoint["scaler"]
        self.imputer = checkpoint["imputer"]
        self.cat_encoder = checkpoint["cat_encoder"]
        self.cat_cardinalities = checkpoint["cat_cardinalities"]
        self.categorical_columns = checkpoint["categorical_columns"]
        self._load_extra_state(checkpoint.get("extra_state", {}))
        self.model = self.build_model(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        return self