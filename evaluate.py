"""Compatibility entrypoint for ``python evaluate.py``."""

from evaluation.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())

