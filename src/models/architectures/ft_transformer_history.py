from typing import List

import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer


class AttentionPooling(nn.Module):
    """Learned additive attention over timesteps. Produces one weighted-sum
    vector per sequence, masked so padded positions get zero weight."""

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # outputs: (B, T, H), mask: (B, T)
        scores = self.score(outputs).squeeze(-1)          # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # rows with zero real timesteps (mask all-zero) would produce all -inf
        all_masked = (mask.sum(dim=1) == 0)
        if all_masked.any():
            scores = scores.clone()
            scores[all_masked] = 0.0

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (outputs * weights).sum(dim=1)                # (B, H)
        return pooled


class CrossAttentionFusion(nn.Module):
    def __init__(self, tab_dim, seq_dim, num_heads=4):
        super().__init__()

        self.query = nn.Linear(tab_dim, seq_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=seq_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, tab_emb, seq_outputs, mask):
        # tab_emb: (B, D_tab)
        # seq_outputs: (B, T, D_seq)

        q = self.query(tab_emb).unsqueeze(1)

        out, _ = self.attn(
            query=q,
            key=seq_outputs,
            value=seq_outputs,
            key_padding_mask=~mask.bool(),
        )

        return out.squeeze(1)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1,): # attn_dim: int = 64
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
#        self.pool = AttentionPooling(hidden_dim, attn_dim=attn_dim)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(seq)          # (B, T, H) — no packing needed,

        return outputs #, mask)     # (B, H)


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
        # seq_attn_dim: int = 64,
    ):
        super().__init__()
        self.tabular_tower = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=tab_embed_dim,   # embedding, not a logit — one linear layer short of FTTransformerClassifier's d_out=1
            **ft_kwargs,
        )
        self.sequence_encoder = SequenceEncoder(
            seq_input_dim, seq_hidden_dim, seq_num_layers, # attn_dim=seq_attn_dim
        )
        self.cross_attention = CrossAttentionFusion( tab_dim=tab_embed_dim, seq_dim=seq_hidden_dim,)
        self.fusion_head = FusionHead(
            tab_embed_dim, seq_hidden_dim, fusion_hidden_dim, fusion_dropout
        )

    def forward(self, x_cont, x_cat, seq, seq_mask):
        # order must match the tensor order built in _make_train_loader:
        # (X_cont, X_cat, *extra_arrays) -> here extra_arrays = (seq, mask)
        tab_emb = self.tabular_tower(x_cont, x_cat)
        seq_outputs = self.sequence_encoder(seq, seq_mask)

        seq_emb = self.cross_attention(
            tab_emb,
            seq_outputs,
            seq_mask,
        )

        return self.fusion_head(tab_emb, seq_emb)