# Troubleshooting

## `CUDA was requested but is unavailable`

In Colab choose a GPU runtime and verify with `nvidia-smi`. Locally install a CUDA-compatible PyTorch wheel. Use `device: auto` for CPU smoke testing.

## CUDA out of memory

PaoPao clears cached CUDA blocks and reports the adjustment path. Set a smaller sequence length in the model config **and** rerun `data.prepare` with the same `--sequence-length`. Keep `gradient_checkpointing: true`, reduce `batch_size`, then increase `gradient_accumulation_steps` to retain an effective batch size.

## Vocabulary mismatch

Train/reload the intended tokenizer and set `model.vocab_size` to the exact value printed by `train_tokenizer`. Do not reuse a token cache made with a different tokenizer.

## Dataset has no full blocks

Add more cleaned text or reduce `--sequence-length`, then rerun `data.prepare`. The metadata file reports retained examples and block counts.

## Checkpoint cannot load

Use only PaoPao's trusted local `.pt` files. A resume checkpoint must have the same `ModelConfig` as the training YAML. For inference, its tokenizer must have the same vocabulary size.

## Command was blocked

PaoPao only executes a restricted single command without a shell. Run repository setup manually outside the agent, or use an isolated development environment for a task that genuinely requires a broader policy.
