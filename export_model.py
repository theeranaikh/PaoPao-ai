"""Export a PaoPao checkpoint and tokenizer for lightweight inference deployment."""

from __future__ import annotations

import argparse

from inference.export import export_for_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a PaoPao inference artifact.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = export_for_inference(args.checkpoint, args.tokenizer, args.output)
    print(f"Exported inference artifact to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
