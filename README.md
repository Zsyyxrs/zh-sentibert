# zh-sentibert

> Read this in: **English** | [简体中文](README.zh-CN.md)

A fine-tuned **BERT-base-Chinese** sentiment classifier for Chinese reviews — trained on **ChnSentiCorp** with selective layer unfreezing, weighted sampling, synonym-based augmentation, mixed-precision training, and warmup-linear LR scheduling.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)

## ✨ Features

- 🇨🇳 **Chinese sentiment classification** (positive / negative) on `lansinuote/ChnSentiCorp` (9.6k train / 1.2k val / 1.2k test).
- 🧊 **Selective fine-tuning** — freeze all BERT layers except the last `N`; add a `Linear → ReLU → Linear` head with Xavier init.
- ⚖️ **Imbalance handling** via `WeightedRandomSampler` and inverse-frequency weights.
- 🔤 **Light text augmentation** — jieba-tokenised synonym replacement at `augment_prob=0.1`.
- ⚡ **Mixed precision training** with `torch.amp.GradScaler` + gradient clipping (`max_norm=1.0`).
- 📉 **Warmup-linear LR schedule** (`get_linear_schedule_with_warmup`).
- 🛑 **Early stopping** on best validation accuracy (patience = 5).
- 📊 **Built-in reporting** — accuracy / precision / recall / F1 / AUC, confusion matrix and training curves saved automatically.
- 🔮 **Inference toolkit** — batch / file / interactive prediction, MC-dropout uncertainty, attention-based explainability.

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/Zsyyxrs/zh-sentibert.git
cd zh-sentibert

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or editable install:
pip install -e .
```

### 2. Train

```bash
python scripts/train.py --epochs 10 --batch_size 32 --lr 2e-5
```

Best checkpoint is written to `checkpoints/best_model_epoch_*.pth`; metrics, curves and the confusion matrix land in `results/`.

### 3. Evaluate

```bash
python scripts/train.py --mode eval --checkpoint checkpoints/best_model_epoch_X.pth
```

### 4. Predict

```bash
# Single text
python scripts/train.py --mode predict --input_text "这家店的氛围真的绝绝子"

# File of texts (one per line)
python scripts/train.py --mode predict --input_file examples/sample_texts.txt

# Interactive REPL
python scripts/train.py --mode predict
```

### 5. Use as a library

```python
from transformers import AutoTokenizer
from zh_sentibert import Config, SentimentPredictor

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
predictor = SentimentPredictor(
    model_path="checkpoints/best_model_epoch_5.pth",
    tokenizer=tokenizer,
    config=Config,
)
print(predictor.predict("电影剧情还可以，就是节奏有点慢。", return_probs=True))
```

## 🏗 Architecture

```
                ┌──────────────────────────┐
   Raw text ──▶ │  DataProcessor.clean()   │  strip HTML / URLs / specials
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │  ImprovedDataset         │  tokenize + synonym aug. (jieba)
                │  WeightedRandomSampler   │  rebalance class frequencies
                └────────────┬─────────────┘
                             ▼
   ┌──────────────────────────────────────────────────┐
   │  bert-base-chinese (12L · 12H · 768d · 110M)     │
   │  ── last N layers + pooler unfrozen ──           │
   └────────────┬─────────────────────────────────────┘
                ▼
   [CLS] pooled  ──▶  Dropout ──▶ Linear(768→768) ──▶ ReLU
                                                       │
                              Dropout ◀────────────────┘
                                  │
                                  ▼
                       Linear(768 → 2)  →  softmax  →  {负向, 正向}

   Training: AdamW · CE loss · warmup-linear LR · AMP · grad-clip · early stop
```

Source layout:

```
zh-sentibert/
├── src/zh_sentibert/        # Importable package
│   ├── config.py            # Hyperparameters + paths
│   ├── data_utils.py        # Dataset, sampler, augmenter, cleaning
│   ├── model.py             # ImprovedModel / MultiTask / Attention variants
│   ├── trainer.py           # Train/val/test loops, metrics, plotting
│   └── inference.py         # SentimentPredictor + interactive REPL
├── scripts/
│   ├── train.py             # CLI entrypoint (train / eval / predict)
│   └── download_dataset.py  # Cache ChnSentiCorp from HF
├── examples/
│   └── sample_texts.txt     # Demo inputs for prediction
├── docs/images/             # Architecture & result screenshots
├── results/                 # Generated metrics, curves, confusion matrices
├── logs/                    # Training logs and history JSON
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## 🛠 Tech Stack

| Layer             | Tooling                                                      |
| ----------------- | ------------------------------------------------------------ |
| Base model        | `google-bert/bert-base-chinese` (12L, 768d, 110M params)   |
| Framework         | PyTorch ≥ 2.0 · Hugging Face `transformers` ≥ 4.30      |
| Data              | `datasets` (`lansinuote/ChnSentiCorp`)                   |
| Chinese tokeniser | `jieba` (for synonym-based augmentation only)              |
| Metrics & plots   | `scikit-learn` · `matplotlib` · `seaborn`            |
| Optimisation      | AdamW ·`get_linear_schedule_with_warmup` · `torch.amp` |
| Experiment log    | `tqdm` + `logging`; optional `wandb`                   |

## 📊 Benchmark / Results

Test set (1200 samples) from ChnSentiCorp, V100 single-GPU, default hyper-params:

| Metric    | Score            |
| --------- | ---------------- |
| Accuracy  | **0.9450** |
| Precision | 0.9451           |
| Recall    | 0.9450           |
| F1        | 0.9450           |
| AUC       | **0.9827** |

<details><summary>Per-class classification report</summary>

```
              precision    recall  f1-score   support
   负向评价     0.9383    0.9510    0.9446       592
   正向评价     0.9517    0.9391    0.9454       608
   accuracy                         0.9450      1200
   macro avg     0.9450    0.9451    0.9450      1200
weighted avg     0.9451    0.9450    0.9450      1200
```

</details>

| Training curves                                   | Confusion matrix                                    |
| ------------------------------------------------- | --------------------------------------------------- |
| ![Training curves](docs/images/training_curves.png) | ![Confusion matrix](docs/images/confusion_matrix.png) |

## 🧪 Troubleshooting

| Symptom                                                           | Cause                                         | Fix                                                                                                             |
| ----------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `Too Many Requests for url: https://hf-mirror.com/api/datasets` | `load_dataset` re-verifies online each call | `export HF_HUB_OFFLINE=1` and pass `download_mode="reuse_dataset_if_exists", verification_mode="no_checks"` |
| CUDA OOM                                                          | Batch too large                               | Lower `--batch_size`, or set `Config.USE_FP16 = True` for inference                                         |

## 🤝 Contributing

Issues and PRs are welcome. Run `ruff` before submitting; if you add a new feature, please include a short test or example. See [LICENSE](LICENSE) for the licensing terms.

## 📄 License

[MIT](LICENSE) © Zsyyxrs
