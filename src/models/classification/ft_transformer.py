from typing import List

import torch.nn as nn
from rtdl_revisiting_models import FTTransformer

from src.models.base_torch_classifier import BaseTorchClassifier


class FTTransformerClassifier(BaseTorchClassifier):

    def __init__(
        self,
        d_token: int = 192,
        n_blocks: int = 3,
        attention_n_heads: int = 8,
        ffn_d_hidden_multiplier: float = 4.0,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        focal_alpha: float = 0.97,
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(focal_alpha=focal_alpha, focal_gamma=focal_gamma, **kwargs,)

        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def build_model(self, input_dim: int) -> nn.Module:

        return FTTransformer(
            n_cont_features=input_dim,
            cat_cardinalities=self.cat_cardinalities,
            d_out=1,
            d_block=self.d_token,
            n_blocks=self.n_blocks,
            attention_n_heads=self.attention_n_heads,
            ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )