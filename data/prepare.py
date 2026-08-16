"""Build cleaned, deduplicated, tokenized PaoPao datasets from local JSONL/Python data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from tokenizer import PaoPaoTokenizer

from .clean import clean_code, content_hash, is_likely_python

LOGGER = logging.getLogger(__name__)


def iter_records(inputs: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Read raw Python files and JSONL records without binding to a dataset provider."""
    for item in inputs:
        path = Path(item)
        files = [path] if path.is_file() else sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        for file_path in files:
            if file_path.suffix == ".py":
                try:
                    yield {"code": file_path.read_text(encoding="utf-8"), "source": str(file_path)}
                except UnicodeDecodeError:
                    continue
            elif file_path.suffix == ".jsonl" or file_path.name.endswith(".jsonl.gz"):
                opener = gzip.open if file_path.name.endswith(".jsonl.gz") else open
                with opener(file_path, "rt", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            LOGGER.warning("Skipping invalid JSON %s:%s", file_path, line_number)
                            continue
                        if isinstance(record, dict):
                            yield record


def _fim(code: str) -> str:
    """Create a deterministic FIM example by moving the middle third to the end."""
    first = len(code) // 3
    second = 2 * len(code) // 3
    return f"<fim_prefix>{code[:first]}<fim_suffix>{code[second:]}<fim_middle>{code[first:second]}"


def format_record(
    record: dict[str, Any], fim_probability: float, completion_probability: float, rng: random.Random
) -> str | None:
    """Map supported record schemas into causal-LM instruction sequences."""
    fixed = record.get("fixed_code") or record.get("fixed")
    buggy = record.get("buggy_code") or record.get("buggy")
    if isinstance(buggy, str) and isinstance(fixed, str):
        buggy, fixed = clean_code(buggy), clean_code(fixed)
        if not buggy or not fixed:
            return None
        error = str(record.get("error") or record.get("error_message") or "Unknown error")
        test_failure = str(record.get("test_failure") or "")
        return (
            f"<|task:bugfix|>\n<|code|>\n{buggy}<|error|>\n{error}\n"
            f"<|test_failure|>\n{test_failure}\n<|code|>\n{fixed}"
        )

    if isinstance(buggy, str):
        buggy = clean_code(buggy)
        if buggy:
            error = str(record.get("error") or record.get("error_message") or "")
            return f"<|task:detect|>\n<|code|>\n{buggy}<|error|>\n{error}"

    prefix, completion = record.get("prefix"), record.get("completion")
    if isinstance(prefix, str) and isinstance(completion, str):
        return f"<|task:complete|>\n<|code|>\n{prefix}{completion}"

    code = clean_code(str(record.get("code") or record.get("function") or ""))
    if not code or not is_likely_python(code):
        return None
    explanation = record.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        return f"<|task:explain|>\n<|code|>\n{code}<|explanation|>\n{explanation.strip()}"
    if rng.random() < fim_probability and len(code) >= 24:
        return _fim(code)
    documentation = record.get("docstring") or record.get("doc") or record.get("description")
    if isinstance(documentation, str) and documentation.strip():
        return f"<|task:generate|>\n{documentation.strip()}\n<|code|>\n{code}"
    if rng.random() < completion_probability and len(code) >= 24:
        split = max(1, len(code) // 2)
        return f"<|task:complete|>\n<|code|>\n{code[:split]}{code[split:]}"
    return f"<|code|>\n{code}"


def pack_examples(examples: Iterable[str], tokenizer: PaoPaoTokenizer, sequence_length: int) -> torch.Tensor:
    """Concatenate examples with EOS and return full, fixed-size training blocks."""
    if sequence_length < 8:
        raise ValueError("sequence_length must be at least 8")
    ids: list[int] = []
    for example in examples:
        ids.extend(tokenizer.encode(example))
        ids.append(int(tokenizer.eos_token_id))
    usable = (len(ids) // sequence_length) * sequence_length
    if usable == 0:
        return torch.empty((0, sequence_length), dtype=torch.long)
    return torch.tensor(ids[:usable], dtype=torch.long).view(-1, sequence_length)


def input_fingerprint(inputs: Iterable[str]) -> str:
    """Fingerprint file names, sizes, and modification times for cache reuse."""
    digest = hashlib.sha256()
    for raw_input in sorted(inputs):
        root = Path(raw_input)
        candidates = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
        for candidate in candidates:
            stats = candidate.stat()
            digest.update(f"{candidate.resolve()}:{stats.st_size}:{stats.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def artifact_fingerprint(path: str | Path) -> str:
    """Hash a tokenizer artifact so a replaced vocabulary invalidates token caches."""
    artifact = Path(path) / "tokenizer.json"
    if not artifact.is_file():
        raise FileNotFoundError(f"tokenizer artifact does not exist: {artifact}")
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


def prepare(
    inputs: list[str],
    tokenizer_path: str,
    output_dir: str,
    sequence_length: int,
    validation_ratio: float,
    fim_probability: float,
    completion_probability: float,
    seed: int,
    force: bool = False,
) -> dict[str, int]:
    """Clean, exact-deduplicate, split, tokenize, and cache local training records."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")
    output = Path(output_dir)
    metadata_path = output / "metadata.json"
    if not 0.0 <= fim_probability <= 1.0 or not 0.0 <= completion_probability <= 1.0:
        raise ValueError("fim_probability and completion_probability must be between zero and one")
    cache_key = hashlib.sha256(
        f"{input_fingerprint(inputs)}:{artifact_fingerprint(tokenizer_path)}:{sequence_length}:{validation_ratio}:{fim_probability}:{completion_probability}:{seed}".encode()
    ).hexdigest()
    if not force and metadata_path.is_file() and (output / "train.pt").is_file() and (output / "validation.pt").is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("cache_key") == cache_key:
            LOGGER.info("Using cached dataset in %s", output)
            return {key: int(value) for key, value in existing.items() if isinstance(value, int)}
    tokenizer = PaoPaoTokenizer.from_file(tokenizer_path)
    rng = random.Random(seed)
    unique: set[str] = set()
    formatted: list[str] = []
    skipped = 0
    for record in iter_records(inputs):
        example = format_record(record, fim_probability, completion_probability, rng)
        if not example:
            skipped += 1
            continue
        digest = content_hash(example)
        if digest in unique:
            continue
        unique.add(digest)
        formatted.append(example)
    rng.shuffle(formatted)
    split_index = max(1, int(len(formatted) * (1.0 - validation_ratio))) if formatted else 0
    train_blocks = pack_examples(formatted[:split_index], tokenizer, sequence_length)
    validation_blocks = pack_examples(formatted[split_index:], tokenizer, sequence_length)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(train_blocks, output / "train.pt")
    torch.save(validation_blocks, output / "validation.pt")
    metadata = {
        "input_fingerprint": input_fingerprint(inputs),
        "cache_key": cache_key,
        "examples": len(formatted),
        "skipped": skipped,
        "train_blocks": int(train_blocks.shape[0]),
        "validation_blocks": int(validation_blocks.shape[0]),
        "sequence_length": sequence_length,
        "vocab_size": tokenizer.vocab_size,
        "seed": seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {key: int(value) for key, value in metadata.items() if isinstance(value, int)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare cached PaoPao training blocks.")
    parser.add_argument("--input", dest="inputs", nargs="+", required=True)
    parser.add_argument("--tokenizer", dest="tokenizer_path", required=True)
    parser.add_argument("--output", dest="output_dir", default="data/processed")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--fim-probability", type=float, default=0.15)
    parser.add_argument("--completion-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Rebuild even when the input cache key matches")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = prepare(**vars(args))
    LOGGER.info("Prepared %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
