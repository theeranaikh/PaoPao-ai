"""Atomic, resumable PaoPao checkpoint I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from model.config import ModelConfig


def save_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    scaler: Any | None,
    step: int,
    epoch: int,
    best_validation_loss: float,
    config: ModelConfig,
    training_config: dict[str, Any],
) -> Path:
    """Atomically save all state needed to resume exactly at the next update."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "model_config": config.to_dict(),
        "training_config": training_config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint, intentionally allowing optimizer objects in trusted local files."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    required = {"model", "step", "epoch", "best_validation_loss", "model_config"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"not a PaoPao training checkpoint; missing {sorted(missing)}")
    return checkpoint


def restore_training_state(
    checkpoint: dict[str, Any],
    model: Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: Any | None = None,
) -> None:
    """Restore model, optimizer, scheduler, scaler, and available RNG state."""
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
