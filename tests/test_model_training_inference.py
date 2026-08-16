from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.dataset import TokenBlockDataset
from inference.generate import LoadedModel, generate
from model import ModelConfig, PaoPaoForCausalLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.trainer import Trainer, TrainingConfig


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 32

    def encode(self, text: str):
        return [3 + (ord(char) % 20) for char in text] or [self.bos_token_id]

    def decode(self, ids):
        return "".join(chr(97 + (token % 26)) for token in ids)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        max_seq_len=16,
        dropout=0.0,
    )


def test_model_forward_and_generation():
    model = PaoPaoForCausalLM(tiny_config()).eval()
    tokens = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (1, 4, 32)
    assert output.loss is not None
    loaded = LoadedModel(model=model, tokenizer=TinyTokenizer(), device=torch.device("cpu"))
    text = generate(loaded, "hi", max_tokens=3, temperature=0.0)
    assert len(text) <= 3


def test_dataset_training_step_and_checkpoint_roundtrip(tmp_path):
    blocks = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]], dtype=torch.long)
    cache = tmp_path / "blocks.pt"
    torch.save(blocks, cache)
    dataset = TokenBlockDataset(cache)
    assert len(dataset) == 2
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = PaoPaoForCausalLM(tiny_config())
    trainer = Trainer(
        model,
        model.config,
        loader,
        loader,
        TrainingConfig(
            output_dir=str(tmp_path / "checkpoints"),
            epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            log_every=1,
            validate_every=1,
            save_every=1,
            mixed_precision=False,
        ),
    )
    trainer.fit()
    latest = tmp_path / "checkpoints" / "latest.pt"
    checkpoint = load_checkpoint(latest)
    assert checkpoint["step"] == 2
    restored = PaoPaoForCausalLM(tiny_config())
    restored.load_state_dict(checkpoint["model"])
    assert restored.parameter_count() == model.parameter_count()

