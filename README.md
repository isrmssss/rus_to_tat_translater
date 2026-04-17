# Russian → Tatar Neural Machine Translation

Transformer-based NMT model trained on 500k Russian-Tatar sentence pairs.

## Model Architecture

| Component | Implementation |
|-----------|----------------|
| Architecture | Transformer (Encoder-Decoder) |
| Position Encoding | Rotary Position Embedding (RoPE) |
| Tokenizer | SentencePiece (BPE, vocab=22000) |
| Norm | Pre-LayerNorm (`norm_first=True`) |
| Regularization | Dropout 0.2, Label Smoothing 0.15 |
| Mixed Precision | AMP |

### Hyperparameters (v2)

```python
d_model = 512
num_layers = 6
dim_feedforward = 2048
batch_size = 24 (effective = 192 with gradient accumulation)
learning_rate = 3e-4
max_len = 96
```

## Files in This Repository

```
rus_to_tat/
├── README.md              # This file
├── .gitignore
├── preprocess.ipynb       # Data preprocessing & tokenization
├── train.ipynb          # Model training
├── tokenizer/
│   └── spm.model     # SentencePiece model (generated)
├── preprocessed/      # Tokenized data (generated)
│   ├── train_tokenized.pt
│   └── test_tokenized.pt
└── checkpoints/v2/   # Model checkpoints (generated)
    ├── model_epoch_5.pt
    ├── optimizer_epoch_5.pt
    ├── scheduler_epoch_5.pt
    └── meta.json
```

## Where to Download

### Dataset (HuggingFace)

```python
from datasets import load_dataset
ds = load_dataset("rmssss/ru-tat-parallel")
ds.save_to_disk("data/")
```

### Pre-trained Model (v2, epoch 5, loss=0.15)

Download from releases or HuggingFace model hub.

## Quick Start

### 1. Clone & Create Empty Folders

```bash
git clone https://github.com/isrmssss/rus_to_tat_translater.git
cd rus_to_tat_translater
mkdir -p data preprocessed tokenizer checkpoints/v2
```

### 2. Download Dataset

```python
from datasets import load_dataset
ds = load_dataset("rmssss/ru-tat-parallel")
ds.save_to_disk("data/")
```

### 3. Preprocessing

```bash
jupyter notebook preprocess.ipynb
```

This generates:
- `tokenizer/spm.model`
- `preprocessed/train_tokenized.pt`
- `preprocessed/test_tokenized.pt`

### 4. Training

```bash
jupyter notebook train.ipynb
```

- Loads latest checkpoint from `checkpoints/v2/` if exists
- Trains for 2 epochs per session (configurable)
- Saves checkpoints after each epoch

### Resume Training

The notebook auto-detects existing checkpoints in `checkpoints/v2/` and resumes automatically.

## Results (v2)

| Epoch | Loss | Global Step |
|-------|------|-------------|
| 5 | **0.15** | 12,955 |

Target: loss < 1.0 ✓ achieved

## Requirements

- Python 3.13+
- PyTorch
- sentencepiece
- pandas
- tqdm
- jupyter