# Training PaoPao

PaoPao initializes every `nn.Linear` and `nn.Embedding` weight randomly. It only reads the tokenizer and dataset files that you create locally.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Prepare a tokenizer and fixed-length token blocks before training:

```bash
python -m tokenizer.train_tokenizer --input data/raw/python --output artifacts/tokenizer --vocab-size 32000
python -m data.prepare --input data/raw/python --tokenizer artifacts/tokenizer \
  --output data/processed --sequence-length 1024
python train.py --config configs/paopao_small.yaml
```

## Google Colab

1. In Colab choose **Runtime > Change runtime type > T4 GPU**.
2. Run the following cells. Replace the repository URL and source-data location.

```bash
!nvidia-smi
!git clone <YOUR_REPOSITORY_URL> PaoPao
%cd PaoPao
!pip install -r requirements.txt
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/paopao/{artifacts,data,checkpoints}
!python -m tokenizer.train_tokenizer \
  --input /content/drive/MyDrive/paopao/data/raw/python \
  --output /content/drive/MyDrive/paopao/artifacts/tokenizer --vocab-size 32000
!python -m data.prepare \
  --input /content/drive/MyDrive/paopao/data/raw/python \
  --tokenizer /content/drive/MyDrive/paopao/artifacts/tokenizer \
  --output /content/drive/MyDrive/paopao/data/processed --sequence-length 512
```

The included `configs/paopao_t4_colab.yaml` already uses those Drive paths. Then train and resume:

```bash
!python train.py --config configs/paopao_t4_colab.yaml
!python train.py --config configs/paopao_t4_colab.yaml \
  --resume /content/drive/MyDrive/paopao/checkpoints/paopao_t4/latest.pt
```

## T4 Memory

The supplied T4 config uses FP16 AMP, sequence length 512, microbatch 1, accumulation 16, and activation checkpointing. A CUDA OOM requires rebuilding the cache with a smaller `--sequence-length` that matches `model.max_seq_len`; do not only change the YAML.

## Checkpoints

`latest.pt` is saved at every epoch and `save_every` update. `best.pt` updates only after a lower validation loss. A checkpoint contains model weights, optimizer/scheduler/scaler states, step/epoch, best loss, configuration, and random-number generator state.

## Export for Inference

Exporting writes a model-only `model.pt` plus the matching locally trained tokenizer files. It is not resumable training state.

```bash
python export_model.py --checkpoint checkpoints/paopao_small/best.pt \
  --tokenizer artifacts/tokenizer --output exports/paopao_small
python inference.py --checkpoint exports/paopao_small/model.pt \
  --tokenizer exports/paopao_small --prompt "Write Python code:"
```
