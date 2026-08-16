"""Export a trained PaoPao checkpoint as a lightweight inference artifact."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from training.checkpoint import load_checkpoint


def export_for_inference(
    checkpoint_path: str | Path, tokenizer_path: str | Path, output_dir: str | Path
) -> Path:
    """Write model-only state and copy the matching locally trained tokenizer."""
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    source_tokenizer, output = Path(tokenizer_path), Path(output_dir)
    if not (source_tokenizer / "tokenizer.json").is_file() or not (source_tokenizer / "tokenizer_config.json").is_file():
        raise FileNotFoundError("tokenizer directory must contain tokenizer.json and tokenizer_config.json")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": "paopao-inference-1",
            "model": checkpoint["model"],
            "model_config": checkpoint["model_config"],
            "source_checkpoint_step": checkpoint["step"],
        },
        output / "model.pt",
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(source_tokenizer / name, output / name)
    return output

