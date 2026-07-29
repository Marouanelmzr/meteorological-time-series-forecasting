from typing import List

import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return h_n[-1]  # (B, hidden_dim) — last layer's final hidden state


class FusionHead(nn.Module):
    def __init__(self, tab_dim: int, seq_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tab_dim + seq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tab_emb: torch.Tensor, seq_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([tab_emb, seq_emb], dim=-1))


class FTTransformerWithHistory(nn.Module):
    def __init__(
        self,
        n_cont_features: int,
        cat_cardinalities: List[int],
        tab_embed_dim: int,
        ft_kwargs: dict,
        seq_input_dim: int,
        seq_hidden_dim: int,
        seq_num_layers: int,
        fusion_hidden_dim: int,
        fusion_dropout: float,
    ):
        super().__init__()
        self.tabular_tower = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=tab_embed_dim,   # embedding, not a logit — one linear layer short of FTTransformerClassifier's d_out=1
            **ft_kwargs,
        )
        self.sequence_encoder = SequenceEncoder(
            seq_input_dim, seq_hidden_dim, seq_num_layers
        )
        self.fusion_head = FusionHead(
            tab_embed_dim, seq_hidden_dim, fusion_hidden_dim, fusion_dropout
        )

    def forward(self, x_cont, x_cat, seq, seq_mask):
        # order must match the tensor order built in _make_train_loader:
        # (X_cont, X_cat, *extra_arrays) -> here extra_arrays = (seq, mask)
        tab_emb = self.tabular_tower(x_cont, x_cat)
        seq_emb = self.sequence_encoder(seq, seq_mask)
        return self.fusion_head(tab_emb, seq_emb)