"""Single-process PyTorch trainer with AMP, accumulation, validation, and resume support."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.config import ModelConfig
from model.modeling_paopao import PaoPaoForCausalLM

from .checkpoint import restore_training_state, save_checkpoint
from .scheduler import build_warmup_cosine_scheduler

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Operational training settings kept separate from model architecture."""

    output_dir: str = "checkpoints"
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    min_lr_scale: float = 0.1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 0
    log_every: int = 10
    validate_every: int = 200
    save_every: int = 200
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("epochs, batch_size, and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")


class Trainer:
    """Train PaoPao without external trainer frameworks or pretrained assets."""

    def __init__(
        self,
        model: PaoPaoForCausalLM,
        model_config: ModelConfig,
        train_loader: DataLoader,
        validation_loader: DataLoader | None,
        config: TrainingConfig,
    ) -> None:
        config.validate()
        self.model = model
        self.model_config = model_config
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95)
        )
        updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        self.total_steps = max(1, updates_per_epoch * config.epochs)
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer, min(config.warmup_steps, self.total_steps - 1), self.total_steps, config.min_lr_scale
        )
        use_amp = config.mixed_precision and self.device.type == "cuda"
        # Prefer the unified API when available, while retaining PyTorch 2.1 support.
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.amp_enabled = use_amp
        self.global_step = 0
        self.start_epoch = 0
        self.best_validation_loss = float("inf")

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device

    def resume(self, path: str) -> None:
        """Restore state and continue from the epoch stored in a local checkpoint."""
        from .checkpoint import load_checkpoint

        checkpoint = load_checkpoint(path, self.device)
        if ModelConfig.from_dict(checkpoint["model_config"]) != self.model_config:
            raise ValueError("checkpoint model configuration does not match the current configuration")
        restore_training_state(checkpoint, self.model, self.optimizer, self.scheduler, self.scaler)
        self.global_step = int(checkpoint["step"])
        self.start_epoch = int(checkpoint["epoch"])
        self.best_validation_loss = float(checkpoint["best_validation_loss"])
        LOGGER.info("Resumed %s at step %s, epoch %s", path, self.global_step, self.start_epoch)

    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _raise_oom_guidance(self, error: RuntimeError) -> None:
        """Release cached CUDA blocks before returning actionable T4 memory guidance."""
        if self.device.type == "cuda" and "out of memory" in str(error).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory: PaoPao cleared the CUDA cache. Reduce training.batch_size or "
                "model.max_seq_len, rebuild token blocks at that sequence length, and increase "
                "gradient_accumulation_steps. Keep gradient_checkpointing enabled on a T4."
            ) from error
        raise error

    @torch.no_grad()
    def validate(self) -> float:
        """Compute mean next-token loss on the validation cache."""
        if self.validation_loader is None or len(self.validation_loader) == 0:
            return float("nan")
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        batches = 0
        for batch in self.validation_loader:
            batch = self._batch_to_device(batch)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                loss = self.model(**batch).loss
            if loss is not None:
                total_loss += loss.item()
                batches += 1
        if was_training:
            self.model.train()
        return total_loss / max(1, batches)

    def _save(self, name: str, epoch: int) -> Path:
        return save_checkpoint(
            Path(self.config.output_dir) / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.global_step,
            epoch,
            self.best_validation_loss,
            self.model_config,
            asdict(self.config),
        )

    def fit(self) -> None:
        """Run training, periodic validation, and best/latest checkpoint persistence."""
        if len(self.train_loader) == 0:
            raise ValueError("training dataset has no full token blocks")
        torch.manual_seed(self.config.seed)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.start_epoch, self.config.epochs):
            progress = tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{self.config.epochs}", leave=False)
            running_loss = 0.0
            for batch_index, batch in enumerate(progress, start=1):
                batch = self._batch_to_device(batch)
                try:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                        output = self.model(**batch)
                        if output.loss is None:
                            raise RuntimeError("model did not return a training loss")
                        loss = output.loss / self.config.gradient_accumulation_steps
                except RuntimeError as exc:
                    self._raise_oom_guidance(exc)
                self.scaler.scale(loss).backward()
                running_loss += loss.item() * self.config.gradient_accumulation_steps
                is_update = (
                    batch_index % self.config.gradient_accumulation_steps == 0 or batch_index == len(self.train_loader)
                )
                if not is_update:
                    continue
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                if self.global_step % self.config.log_every == 0:
                    LOGGER.info(
                        "step=%s loss=%.4f lr=%.3e", self.global_step, running_loss / max(1, batch_index),
                        self.scheduler.get_last_lr()[0]
                    )
                if self.validation_loader and self.global_step % self.config.validate_every == 0:
                    validation_loss = self.validate()
                    LOGGER.info("step=%s validation_loss=%.4f perplexity=%.2f", self.global_step, validation_loss, math.exp(min(validation_loss, 20)))
                    if validation_loss < self.best_validation_loss:
                        self.best_validation_loss = validation_loss
                        self._save("best.pt", epoch)
                if self.global_step % self.config.save_every == 0:
                    self._save("latest.pt", epoch)
            validation_loss = self.validate()
            if not math.isnan(validation_loss) and validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self._save("best.pt", epoch + 1)
            self._save("latest.pt", epoch + 1)
            LOGGER.info("finished epoch=%s validation_loss=%s", epoch + 1, validation_loss)
