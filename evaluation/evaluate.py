"""Evaluate loss, syntax, optional test execution, and repair exact-match outcomes."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import TokenBlockDataset
from inference.generate import generate, load_model
from tools.test_runner import run_tests


@torch.inference_mode()
def validation_metrics(loaded: Any, dataset_path: str, batch_size: int = 2) -> dict[str, float]:
    """Calculate mean token loss and perplexity for cached validation blocks."""
    dataset = TokenBlockDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size)
    if len(loader) == 0:
        return {"validation_loss": float("nan"), "perplexity": float("nan")}
    total, count = 0.0, 0
    for batch in loader:
        batch = {key: value.to(loaded.device) for key, value in batch.items()}
        loss = loaded.model(**batch).loss
        if loss is not None:
            total += loss.item()
            count += 1
    loss = total / max(1, count)
    return {"validation_loss": loss, "perplexity": math.exp(min(loss, 20))}


def syntax_correctness(loaded: Any, prompts_path: str, max_tokens: int) -> dict[str, float]:
    """Generate continuations for text prompts and check whether the joined Python parses."""
    prompts = [line for line in Path(prompts_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        return {"generation_count": 0.0, "syntax_correctness": float("nan")}
    successful = 0
    for prompt in prompts:
        candidate = prompt + generate(loaded, prompt, max_tokens=max_tokens, temperature=0.0)
        try:
            ast.parse(candidate)
            successful += 1
        except SyntaxError:
            pass
    return {"generation_count": float(len(prompts)), "syntax_correctness": successful / len(prompts)}


def bugfix_success_rate(loaded: Any, repair_path: str, max_tokens: int) -> dict[str, float]:
    """Measure exact normalized fixed-code completion on repair JSONL examples."""
    total, successful = 0, 0
    with Path(repair_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            buggy = record.get("buggy_code") or record.get("buggy")
            fixed = record.get("fixed_code") or record.get("fixed")
            if not isinstance(buggy, str) or not isinstance(fixed, str):
                continue
            prompt = f"<|task:bugfix|>\n<|code|>\n{buggy}\n<|error|>\n{record.get('error', '')}\n<|code|>\n"
            generated = generate(loaded, prompt, max_tokens=max_tokens, temperature=0.0)
            total += 1
            if generated.strip() == fixed.strip():
                successful += 1
    return {"bugfix_examples": float(total), "bugfix_success_rate": successful / total if total else float("nan")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a local PaoPao checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--validation-data")
    parser.add_argument("--prompts", help="One Python prefix prompt per line")
    parser.add_argument("--bugfix-data", help="Repair JSONL with buggy_code/fixed_code/error")
    parser.add_argument("--test-path", help="Project path whose tests should be executed")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    loaded = load_model(args.checkpoint, args.tokenizer, args.device)
    metrics: dict[str, object] = {}
    if args.validation_data:
        metrics.update(validation_metrics(loaded, args.validation_data))
    if args.prompts:
        metrics.update(syntax_correctness(loaded, args.prompts, args.max_tokens))
    if args.bugfix_data:
        metrics.update(bugfix_success_rate(loaded, args.bugfix_data, args.max_tokens))
    if args.test_path:
        result = run_tests(args.test_path)
        metrics["unit_tests_passed"] = result.returncode == 0
        metrics["unit_test_returncode"] = result.returncode
    print(json.dumps(metrics, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

