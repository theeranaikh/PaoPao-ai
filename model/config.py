"""Configuration for the PaoPao decoder-only Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Hyperparameters for a newly initialized PaoPao language model."""

    vocab_size: int = 32_000
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    intermediate_size: int = 2_048
    max_seq_len: int = 1_024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("vocab_size, hidden_size, and num_layers must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.intermediate_size < self.hidden_size:
            raise ValueError("intermediate_size must be at least hidden_size")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        """Create a configuration while rejecting unknown checkpoint fields."""
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})

