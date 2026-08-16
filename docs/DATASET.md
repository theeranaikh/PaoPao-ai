# Dataset Pipeline

## Sources

Use source code you are licensed to process. `python -m data.download` explicitly downloads the CodeSearchNet Python archive URL and can extract it safely. `data.prepare` accepts directories of `.py` files, `.jsonl` files, and CodeSearchNet `.jsonl.gz` shards.

```bash
python -m data.download --output data/raw/codesearchnet-python.zip \
  --extract-dir data/raw/codesearchnet-python
```

## JSONL Schemas

Generation records can use:

```json
{"code": "def add(a, b):\n    return a + b\n", "docstring": "Add two values."}
```

Repair records use:

```json
{"buggy_code": "def ratio(a, b):\n    return a / b\n", "error": "ZeroDivisionError", "test_failure": "test_ratio_zero failed", "fixed_code": "def ratio(a, b):\n    if b == 0:\n        return None\n    return a / b\n"}
```

`fixed`/`buggy`, `doc`, and `description` are accepted aliases. Invalid JSON, empty source, NUL bytes, samples longer than 200,000 characters, and exact duplicate formatted examples are removed.

## Preparation

```bash
python -m data.prepare --input data/raw/python repairs.jsonl \
  --tokenizer artifacts/tokenizer --output data/processed \
  --sequence-length 1024 --validation-ratio 0.02 --fim-probability 0.15
```

The output is `train.pt`, `validation.pt`, and `metadata.json`. Each `.pt` cache is a rank-2 tensor with full blocks of token IDs. An unchanged input fingerprint and preparation settings reuse the existing cache; use `--force` to rebuild. Regenerate it after changing tokenizer, corpus, or sequence length. PaoPao supports generation, completion, explanation, bug detection, bug repair, error-to-repair, and FIM sequences through its local task tokens.
