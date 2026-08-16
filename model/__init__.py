"""PaoPao's from-scratch decoder-only Transformer."""

from .config import ModelConfig
from .modeling_paopao import PaoPaoForCausalLM

__all__ = ["ModelConfig", "PaoPaoForCausalLM"]

