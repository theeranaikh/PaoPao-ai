# Bug Fixing Workflow

Use a trained checkpoint:

```bash
paopao --checkpoint checkpoints/paopao_small/best.pt --tokenizer artifacts/tokenizer \
  --root /path/to/project fix main.py
```

1. PaoPao reads and compiles the requested file, then runs it under limits.
2. It selects the target, imported sibling modules, relevant test files, and error-related filenames within the workspace.
3. The local model receives the bounded context and must return complete replacement source in a Python code fence.
4. PaoPao renders a unified diff. No filesystem write happens at this stage.
5. The user accepts with `y`, or passes `--apply` for an explicit noninteractive approval.
6. It applies the exact reviewed patch, runs the target and project test suite, and retains it only on success.
7. A failed patch is rolled back. PaoPao can regenerate and retry up to `--retries + 1` attempts.

Without `--checkpoint`, `fix` can diagnose but cannot invent a code change. `explain` still gives a static syntax/function/class summary without a model.

