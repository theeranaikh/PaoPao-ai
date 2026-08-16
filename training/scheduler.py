"""Learning-rate scheduling helpers."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr_scale: float = 0.1
) -> LambdaLR:
    """Create a step scheduler with linear warmup followed by cosine decay."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be non-negative and smaller than total_steps")
    if not 0.0 <= min_lr_scale <= 1.0:
        raise ValueError("min_lr_scale must be in [0, 1]")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, schedule)

