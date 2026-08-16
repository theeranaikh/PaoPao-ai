"""Entrypoint for training PaoPao from randomly initialized weights."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import TokenBlockDataset
from model import ModelConfig, PaoPaoForCausalLM
from tokenizer import PaoPaoTokenizer
from training.trainer import Trainer, TrainingConfig


def load_yaml(path: str) -> dict[str, Any]:
    """Read a mapping-only YAML file."""
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("training configuration must be a YAML mapping")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PaoPao from scratch.")
    parser.add_argument("--config", default="configs/paopao_small.yaml")
    parser.add_argument("--resume", help="Trusted local PaoPao checkpoint to resume")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = load_yaml(args.config)
    tokenizer = PaoPaoTokenizer.from_file(raw["tokenizer_path"])
    model_values = dict(raw["model"])
    configured_vocab = model_values.pop("vocab_size", tokenizer.vocab_size)
    if configured_vocab != tokenizer.vocab_size:
        raise ValueError("model.vocab_size must equal the trained tokenizer vocabulary size")
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **model_values)
    training_config = TrainingConfig(**raw["training"])
    data_config = raw["data"]
    train_dataset = TokenBlockDataset(data_config["train_path"])
    validation_path = data_config.get("validation_path")
    validation_dataset = TokenBlockDataset(validation_path) if validation_path else None
    loader_args = {"batch_size": training_config.batch_size, "num_workers": training_config.num_workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args) if validation_dataset else None
    model = PaoPaoForCausalLM(model_config)
    logging.getLogger(__name__).info("PaoPao parameters: %s", f"{model.parameter_count():,}")
    trainer = Trainer(model, model_config, train_loader, validation_loader, training_config)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

