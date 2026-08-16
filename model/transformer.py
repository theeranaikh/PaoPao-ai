"""Transformer blocks for PaoPao."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import CausalSelfAttention
from .config import ModelConfig


class FeedForward(nn.Module):
    """GPT-style MLP with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(hidden_states), approximate="tanh")))


class TransformerBlock(nn.Module):
    """Pre-normalized causal Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

