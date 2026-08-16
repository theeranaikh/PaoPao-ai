# Architecture

PaoPao is a decoder-only Transformer implemented in `model/`. It has token and learned positional embeddings, pre-LayerNorm blocks, causal multi-head self-attention, GELU feed-forward networks, residual connections, final LayerNorm, and a language-model head. The head can share the token embedding matrix.

`ModelConfig` owns `vocab_size`, `hidden_size`, `num_layers`, `num_heads`, `intermediate_size`, `max_seq_len`, `dropout`, and operational architectural options. `configs/paopao_small.yaml` selects 12 layers, hidden size 512, 8 heads, MLP size 2048, and a 32,000-token vocabulary. With tied embeddings it is about 54M trainable parameters.

The model has no pretrained loading code. `PaoPaoForCausalLM` initializes linear and embedding weights with a normal distribution (`std=0.02`), and only `training.checkpoint.load_checkpoint` loads weights produced by a previous local PaoPao run.

To scale the architecture, edit model dimensions and train a matching tokenizer/cache. Ensure `hidden_size % num_heads == 0`, ensure the cache sequence length is not above `max_seq_len`, and budget GPU memory for activations.

