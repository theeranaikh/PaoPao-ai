# PaoPao

PaoPao is a Python-first coding LLM and local coding agent. Its decoder-only Transformer and Byte-Level BPE tokenizer are trained **from scratch** with PyTorch. It does not download, load, or inherit pretrained model weights or tokenizers.

## Features

- Configurable 54M-parameter PaoPao Small decoder-only Transformer.
- Locally trained code tokenizer preserving whitespace, indentation, operators, strings, comments, and error text.
- Data preparation for raw Python, CodeSearchNet-style JSONL, repair JSONL, and Fill-in-the-Middle examples.
- AMP, gradient accumulation/clipping, AdamW, warmup/cosine decay, validation, checkpoints, and resume.
- Local generation and evaluation for loss, perplexity, Python syntax, optional tests, and repair exact match.
- CLI coding agent with scoped context, sandboxed commands, reviewable diffs, rollback, and bounded repair retries.

## Requirements

- Python 3.10+
- PyTorch 2.1+ (CUDA build for NVIDIA GPU training)
- Disk space for the source corpus, tokenized cache, and checkpoints

## Installation

```bash
git clone <YOUR_REPOSITORY_URL> PaoPao
cd PaoPao
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```text
model/          Decoder-only Transformer and configuration
tokenizer/      Fresh Byte-Level BPE training and loading
data/           Downloading, cleaning, preparation, cached datasets
training/       Trainer, scheduler, checkpoints
inference/      Checkpoint loading and token sampling
evaluation/     Loss, syntax, repair, and test evaluation
agent/          Context selection, tool controller, repair workflow
tools/          Command sandbox, filesystem patches, Python/test runners
cli/            paopao command-line entrypoint
configs/         Small and NVIDIA T4 configurations
docs/            Operational documentation
tests/           Smoke tests
```

## Dataset Preparation

PaoPao accepts raw `.py` files and JSONL. A repair record uses `buggy_code`, `fixed_code`, and optional `error`/`test_failure` fields.

```bash
python -m data.download --output data/raw/codesearchnet-python.zip \
  --extract-dir data/raw/codesearchnet-python
python -m tokenizer.train_tokenizer --input data/raw/python --output artifacts/tokenizer --vocab-size 32000
python -m data.prepare --input data/raw/python repairs.jsonl \
  --tokenizer artifacts/tokenizer --output data/processed --sequence-length 1024
```

See [docs/DATASET.md](docs/DATASET.md) for supported JSONL schemas and cache output.

## Tokenizer Training

```bash
python -m tokenizer.train_tokenizer \
  --input data/raw/python repairs.jsonl \
  --output artifacts/tokenizer --vocab-size 32000 --min-frequency 2
```

This command makes `artifacts/tokenizer/tokenizer.json` from the supplied files. It does not use a GPT-2, LLaMA, Qwen, or other pretrained tokenizer.

## Training From Scratch

```bash
python train.py --config configs/paopao_small.yaml
```

The tokenizer's actual vocabulary size must match `model.vocab_size`. Checkpoints are written to `checkpoints/paopao_small/latest.pt` and, after validation improves, `best.pt`.

## Training on Google Colab

Follow the copyable Colab sequence in [docs/TRAINING.md](docs/TRAINING.md). It mounts Google Drive before training so tokenized data and checkpoints persist between sessions.

## NVIDIA T4 Configuration

```bash
python train.py --config configs/paopao_t4.yaml
```

`paopao_t4.yaml` uses a 512-token sequence length, FP16 AMP, batch size 1, 16 accumulation steps, and gradient checkpointing for a 16 GB T4. When it OOMs, lower `max_seq_len` in both configuration and dataset preparation, then increase accumulation to preserve effective batch size.

## Resume Training

```bash
python train.py --config configs/paopao_small.yaml \
  --resume checkpoints/paopao_small/latest.pt
```

The checkpoint includes model, optimizer, scheduler, AMP scaler, epoch, step, best validation loss, configs, and RNG states. Only resume trusted local checkpoint files.

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/paopao_small/best.pt \
  --tokenizer artifacts/tokenizer \
  --validation-data data/processed/validation.pt \
  --prompts eval_prompts.txt \
  --bugfix-data repairs.jsonl \
  --test-path .
```

It emits only metrics for supplied inputs: validation loss/perplexity, parse rate for generated Python continuations, exact normalized repair completion rate, and the project test result.

## Inference

```bash
python inference.py \
  --checkpoint checkpoints/paopao_small/best.pt \
  --tokenizer artifacts/tokenizer \
  --prompt "<|task:generate|>\nWrite a Python Fibonacci function.\n<|code|>\n" \
  --temperature 0.2 --top-k 50 --top-p 0.95 --max-tokens 200
```

To export a model-only deployment artifact with its matching tokenizer:

```bash
python export_model.py --checkpoint checkpoints/paopao_small/best.pt \
  --tokenizer artifacts/tokenizer --output exports/paopao_small
```

## CLI Usage

```bash
paopao --checkpoint checkpoints/paopao_small/best.pt --tokenizer artifacts/tokenizer \
  generate "Write a Python Fibonacci function"
paopao --root . run main.py
paopao --root . explain main.py
paopao --root . test
paopao --checkpoint checkpoints/paopao_small/best.pt --tokenizer artifacts/tokenizer \
  --root . fix main.py
```

`paopao chat` needs the same `--checkpoint` and `--tokenizer` options. `paopao "prompt"` is accepted as shorthand for `paopao generate "prompt"` when it is the first argument; prefer the explicit form in scripts.

## Terminal Agent

The agent exposes `read_file`, `list_files`, `execute_command`, `run_python`, `run_tests`, `write_file`, and `edit_file` through an allowlisted controller. Commands execute without a shell and have timeout, CPU, address-space, output, workspace, and network-command restrictions. See [docs/CLI.md](docs/CLI.md).

## Automatic Bug Fixing

```bash
paopao --checkpoint checkpoints/paopao_small/best.pt --tokenizer artifacts/tokenizer \
  --root . fix main.py
```

The command diagnoses the file, selects related project context, and displays a unified diff. Enter `y` to approve it, or use `--apply` in a noninteractive workflow. The agent runs the file and tests after each approved attempt, rolls back a failing patch, and makes at most `--retries + 1` attempts.

## Testing

```bash
pytest -q
python3 -m compileall -q .
```

The smoke suite covers tokenizer training/loading, model forward, cached dataset loading, one training update, checkpoint round trip, generation, CLI parsing, terminal execution, and the no-write repair workflow. Tests requiring `torch` or `tokenizers` skip with a clear reason when dependencies are not installed.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for CUDA OOM, tokenizer-vocabulary mismatch, empty caches, checkpoints, and command-policy failures.

## Safety

PaoPao never grants administrator privileges. It blocks destructive and common network/package commands, invokes no shell, constrains resources, and keeps file access under `--root`. Review every diff. The built-in command policy is not an isolation boundary for malicious Python; run untrusted code in a dedicated container or VM with OS-enforced filesystem and network isolation.

## License

MIT. See [LICENSE](LICENSE).
