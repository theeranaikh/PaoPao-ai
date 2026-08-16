"""Generate code from a trained local PaoPao checkpoint."""

from __future__ import annotations

import argparse

from inference.generate import generate, load_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate with a trained PaoPao checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--stop-token", action="append", type=int, default=[])
    parser.add_argument("--stop-string", action="append", default=[])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    loaded = load_model(args.checkpoint, args.tokenizer, args.device)
    result = generate(
        loaded,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_tokens=args.stop_token or None,
        stop_strings=args.stop_string or None,
    )
    print(result, end="" if result.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

