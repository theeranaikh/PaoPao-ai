# CLI

Install the command with `pip install -e .`, then run from a trusted project root.

```bash
paopao --checkpoint checkpoints/best.pt --tokenizer artifacts/tokenizer chat
paopao --checkpoint checkpoints/best.pt --tokenizer artifacts/tokenizer generate "Create a FastAPI route"
paopao --root . run main.py
paopao --root . explain main.py
paopao --root . test
paopao --checkpoint checkpoints/best.pt --tokenizer artifacts/tokenizer --root . fix main.py
```

`fix` prints the diagnosis and diff. It asks for an interactive accept/reject decision. `--apply` is an explicit approval for the bounded repair session and is intended for CI-like scripts. `--retries N` limits retries after failed verification.

Terminal commands have no shell, run beneath `--root`, and are limited by timeout, CPU, virtual memory, and output length. Destructive commands, package management, privilege escalation, disk administration, and common network clients are blocked. This policy cannot make arbitrary Python code safe; use a container/VM for untrusted repositories.

