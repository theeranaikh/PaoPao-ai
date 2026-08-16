from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from inference.export import export_for_inference
from inference.generate import load_model
from model import ModelConfig, PaoPaoForCausalLM
from tokenizer import PaoPaoTokenizer
from training.checkpoint import save_checkpoint


def test_exported_artifact_loads_for_inference(tmp_path):
    tokenizer = PaoPaoTokenizer.train(["def add(a, b):\n    return a + b\n"], vocab_size=300, min_frequency=1)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        max_seq_len=16,
        dropout=0.0,
    )
    model = PaoPaoForCausalLM(config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, None, None, None, 1, 1, 1.0, config, {})
    artifact_dir = export_for_inference(checkpoint, tokenizer_dir, tmp_path / "export")
    loaded = load_model(artifact_dir / "model.pt", artifact_dir, device="cpu")
    assert loaded.model.config == config
    assert loaded.tokenizer.vocab_size == tokenizer.vocab_size
