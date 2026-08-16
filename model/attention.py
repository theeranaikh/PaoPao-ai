"""Causal multi-head attention used by PaoPao."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Masked self-attention with explicit causal masking for broad PyTorch support."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply causal attention to ``[batch, sequence, hidden]`` states."""
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(
            sequence_length, sequence_length, device=hidden_states.device, dtype=torch.bool
        ).tril_()
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = self.attention_dropout(probabilities)
        attended = probabilities @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        return self.residual_dropout(self.output(attended))

