"""CLI for training PaoPao's tokenizer from local code corpora."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator

from .tokenizer import PaoPaoTokenizer


def iter_corpus(paths: list[str]) -> Iterator[str]:
    """Yield UTF-8 source files, JSONL text entries, and corpus text files."""
    for raw_path in paths:
        path = Path(raw_path)
        candidates = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
        for candidate in candidates:
            is_jsonl_gzip = candidate.name.endswith(".jsonl.gz")
            if candidate.suffix.lower() not in {".py", ".txt", ".md", ".jsonl"} and not is_jsonl_gzip:
                continue
            try:
                if candidate.suffix == ".jsonl" or is_jsonl_gzip:
                    opener = gzip.open if is_jsonl_gzip else open
                    with opener(candidate, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            record = json.loads(line)
                            if isinstance(record, dict):
                                for field in ("code", "function", "buggy_code", "fixed_code", "docstring", "description"):
                                    value = record.get(field)
                                    if isinstance(value, str) and value.strip():
                                        yield value
                    continue
                content = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except (OSError, json.JSONDecodeError):
                continue
            if content.strip():
                yield content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a fresh PaoPao code tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Files or directories containing code/text")
    parser.add_argument("--output", default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args(argv)
    tokenizer = PaoPaoTokenizer.train(
        iter_corpus(args.input), vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    tokenizer.save(args.output)
    print(f"Saved tokenizer with {tokenizer.vocab_size} tokens to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
