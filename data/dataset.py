"""Cached fixed-length token blocks for causal-language-model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


def torch_load(path: str | Path) -> Any:
    """Load local dataset/checkpoint tensors across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TokenBlockDataset(Dataset[dict[str, Tensor]]):
    """A tensor-backed cache where every row is a complete fixed-length example."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        blocks = torch_load(self.path)
        if not isinstance(blocks, Tensor) or blocks.ndim != 2:
            raise ValueError(f"{self.path} must contain a rank-2 tensor of token IDs")
        self.blocks = blocks.long()

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        block = self.blocks[index]
        return {"input_ids": block, "labels": block.clone()}

