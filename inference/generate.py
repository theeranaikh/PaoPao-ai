"""Autoregressive text generation from a local PaoPao checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F

from model import ModelConfig, PaoPaoForCausalLM
from tokenizer import PaoPaoTokenizer
from training.checkpoint import load_checkpoint


@dataclass
class LoadedModel:
    """A model and matching tokenizer loaded from local artifacts."""

    model: PaoPaoForCausalLM
    tokenizer: PaoPaoTokenizer
    device: torch.device


def load_model(
    checkpoint_path: str | Path, tokenizer_path: str | Path, device: str = "auto"
) -> LoadedModel:
    """Load a trained local checkpoint; no remote model or weights are consulted."""
    selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    try:
        checkpoint = load_checkpoint(checkpoint_path, selected_device)
    except ValueError:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=selected_device)
        if checkpoint.get("format_version") != "paopao-inference-1":
            raise
        if not {"model", "model_config"}.issubset(checkpoint):
            raise ValueError("not a PaoPao inference artifact")
    model = PaoPaoForCausalLM(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(selected_device).eval()
    tokenizer = PaoPaoTokenizer.from_file(tokenizer_path)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match the supplied tokenizer")
    return LoadedModel(model=model, tokenizer=tokenizer, device=selected_device)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k > 0:
        keep = min(top_k, logits.numel())
        threshold = torch.topk(logits, keep).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = F.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.scatter(0, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


@torch.inference_mode()
def generate(
    loaded: LoadedModel,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_k: int = 50,
    top_p: float = 0.95,
    stop_tokens: Sequence[int] | None = None,
    stop_strings: Sequence[str] | None = None,
) -> str:
    """Generate continuation text using a bounded context window and sampling controls."""
    if max_tokens <= 0:
        return ""
    input_ids = loaded.tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [int(loaded.tokenizer.bos_token_id)]
    generated: list[int] = []
    stops = set(stop_tokens or [int(loaded.tokenizer.eos_token_id)])
    for _ in range(max_tokens):
        context = (input_ids + generated)[-loaded.model.config.max_seq_len :]
        tensor = torch.tensor([context], dtype=torch.long, device=loaded.device)
        logits = loaded.model(tensor).logits[0, -1]
        token = _sample(logits, temperature, top_k, top_p)
        if token in stops:
            break
        generated.append(token)
        text = loaded.tokenizer.decode(generated)
        if stop_strings and any(stop in text for stop in stop_strings):
            for stop in stop_strings:
                text = text.split(stop, 1)[0]
            return text
    return loaded.tokenizer.decode(generated)
