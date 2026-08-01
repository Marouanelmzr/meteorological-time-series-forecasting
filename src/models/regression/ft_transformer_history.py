from typing import Optional

import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.models.architectures.ft_transformer_history import FTTransformerWithHistory
from src.models.base_torch_regressor import BaseTorchRegressor


class FTTransformerWithHistoryRegressor(BaseTorchRegressor):

    def __init__(
        self,
        d_token: int = 192,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        ffn_d_hidden_multiplier: float = 4.0,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        tab_embed_dim: int = 64,
        seq_input_dim: int = 6,          # wind, temp, pressure, wind_dir_sin, wind_dir_cos, hours_before_run
        seq_hidden_dim: int = 64,
        seq_num_layers: int = 1,
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.1,
        loss_fn: str = "smooth_l1",
        **kwargs,
    ):
        super().__init__(loss_fn=loss_fn, **kwargs,)

        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.tab_embed_dim = tab_embed_dim
        self.seq_input_dim = seq_input_dim
        self.seq_hidden_dim = seq_hidden_dim
        self.seq_num_layers = seq_num_layers
        self.fusion_hidden_dim = fusion_hidden_dim
        self.fusion_dropout = fusion_dropout
        self.loss_fn = loss_fn

        self.seq_imputer: Optional[SimpleImputer] = None
        self.seq_scaler: Optional[StandardScaler] = None

    def build_model(self, input_dim: int) -> nn.Module:

        return FTTransformerWithHistory(
            n_cont_features=input_dim,
            cat_cardinalities=self.cat_cardinalities,
            tab_embed_dim=self.tab_embed_dim,
            ft_kwargs=dict(
                d_block=self.d_token,
                n_blocks=self.n_blocks,
                attention_n_heads=self.attention_n_heads,
                ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
                attention_dropout=self.attention_dropout,
                ffn_dropout=self.ffn_dropout,
                residual_dropout=self.residual_dropout,
            ),
            seq_input_dim=self.seq_input_dim,
            seq_hidden_dim=self.seq_hidden_dim,
            seq_num_layers=self.seq_num_layers,
            fusion_hidden_dim=self.fusion_hidden_dim,
            fusion_dropout=self.fusion_dropout,
        )

    def _prepare_extra_inputs(self, extra, fit: bool = False):
        if extra is None:
            return ()

        seq, mask = extra
        N, T, F = seq.shape
        flat = seq.reshape(-1, F)
        flat_mask = mask.reshape(-1).astype(bool)

        real = flat[flat_mask]

        if fit:
            self.seq_imputer = SimpleImputer(strategy="median")
            real_imputed = self.seq_imputer.fit_transform(real)

            self.seq_scaler = StandardScaler()
            self.seq_scaler.fit(real_imputed)
        else:
            real_imputed = self.seq_imputer.transform(real)

        if self.seq_scaler is None or self.seq_imputer is None:
            raise RuntimeError(
                "seq_imputer/seq_scaler not fitted — call fit() before predict()/transform()."
            )

        real_scaled = self.seq_scaler.transform(real_imputed)

        flat_scaled = flat.copy()
        flat_scaled[flat_mask] = real_scaled
        flat_scaled[~flat_mask] = 0.0  # padded positions -> 0, mask carries the rest

        seq_scaled = flat_scaled.reshape(N, T, F).astype(np.float32)
        return (seq_scaled, mask.astype(np.float32))

    def _extra_state(self) -> dict:
        return {"seq_imputer": self.seq_imputer, "seq_scaler": self.seq_scaler}

    def _load_extra_state(self, state: dict) -> None:
        self.seq_imputer = state.get("seq_imputer")
        self.seq_scaler = state.get("seq_scaler")