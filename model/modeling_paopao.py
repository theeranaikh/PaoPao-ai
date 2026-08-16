"""PaoPao causal language model, initialized from scratch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .transformer import TransformerBlock


@dataclass
class CausalLMOutput:
    """Outputs returned by :class:`PaoPaoForCausalLM`."""

    logits: Tensor
    loss: Optional[Tensor] = None


class PaoPaoForCausalLM(nn.Module):
    """A configurable decoder-only code language model with fresh parameters."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> CausalLMOutput:
        """Return logits and, when labels are supplied, next-token cross-entropy loss."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max_seq_len {self.config.max_seq_len}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        hidden_states = self.embedding_dropout(
            self.token_embeddings(input_ids) + self.position_embeddings(positions)
        )
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    def parameter_count(self, trainable_only: bool = True) -> int:
        """Return the number of model parameters without double-counting tied weights."""
        seen: set[int] = set()
        count = 0
        for parameter in self.parameters():
            if (not trainable_only or parameter.requires_grad) and id(parameter) not in seen:
                seen.add(id(parameter))
                count += parameter.numel()
        return count

