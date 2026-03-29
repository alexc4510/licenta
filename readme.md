# AI-Generated Image Detection

A research pipeline for binary classification of AI-generated vs. real images using convolutional neural networks, with a focus on cross-dataset generalisation.

The core research question is whether CNN-based detectors learn genuine generative artefacts that transfer across datasets and generator families, or whether they overfit to dataset-specific shortcuts. To answer this, models are trained on one dataset and evaluated against all others, producing a generalisation matrix for each architecture.

---

## Datasets

| ID | Name | Source | Images | Generator | Task |
|----|------|--------|--------|-----------|------|
| `ai_vs_human` | AI vs Human Generated Dataset | [Kaggle](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) | ~80k | Diffusion (SD, MidJourney, DALL-E) | Binary (real / AI) |
| `dataset_b` | Defactify Image Dataset | [HuggingFace](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) | 96k | Diffusion (SD 2.1, SDXL, SD3, DALL-E 3, MidJourney v6) | Binary (real / AI) |
| `dataset_c` | 140k Real and Fake Faces | [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | 140k | GAN (StyleGAN) | Binary (real / AI) |

`ai_vs_human` contains real images from Shutterstock paired with AI-generated counterparts produced by DeepMedia. `dataset_b` is caption-aligned (MS-COCO captions), with each real image paired with five synthetic versions from modern diffusion models. `dataset_c` contains real faces from Flickr-Faces-HQ (NVIDIA) paired with StyleGAN-generated faces.

---

## Models

| Model | Architecture | Parameters |
|-------|-------------|------------|
| `resnet18` | ResNet-18, pretrained ImageNet | ~11.7M |
| `resnet50` | ResNet-50, pretrained ImageNet | ~25.6M |
| `vit` | ViT-B/16, pretrained ImageNet | ~86M |

---

## Project structure

```
licenta/
├── .gitignore
├── requirements.txt
├── README.md
│
├── scripts/                         # data download and preparation
│   ├── download_dataset_a.py        # fetch ai_vs_human from Kaggle → Drive
│   ├── prepare_dataset_a.py         # split + resize → datasets/ai_vs_human/
│   ├── download_dataset_b.py        # fetch Defactify from HuggingFace → Drive
│   ├── prepare_dataset_b.py         # verify + manifest → datasets/dataset_b/
│   └── download_dataset_c.py        # fetch 140k faces from Kaggle → Drive
│
└── src/
    ├── config.py                    # all paths, hyperparameters
    ├── models/
    │   ├── resnet18.py
    │   ├── resnet50.py
    │   └── vit.py                   # ViT-B/16
    ├── data/
    │   └── dataset.py               # DataLoader factory, transforms
    ├── training/
    │   ├── _common.py               # shared loop, checkpoint save/load, plots
    │   ├── train_resnet18.py
    │   └── train_resnet50.py
    └── evaluation/
        ├── _common.py               # shared inference + metrics logic
        ├── evaluate_resnet18.py     # single-model test evaluation
        ├── evaluate_resnet50.py
        └── cross_dataset_matrix.py  # full N×N generalisation matrix
```

All data, checkpoints, and outputs are stored on Google Drive and are **not** tracked by git (see `.gitignore`). The Drive layout mirrors the dataset keys in `config.py`:

```
MyDrive/licenta/
├── datasets/
│   ├── ai_vs_human/   train/  val/  test/   (each with ai/ and real/)
│   ├── dataset_b/     train/  val/  test/   + manifest.csv
│   └── dataset_c/     train/  val/  test/
├── checkpoints/
│   └── <dataset>/<model>/   resnet50_epoch_NN.pth  ...
└── logs/
    ├── <dataset>/<model>/   metrics CSV + loss/accuracy plots
    └── cross_dataset_matrix/<model>/<NxN>/   accuracy_matrix.csv + .png  ...
```

---

## Setup

### Prerequisites

- Google account with sufficient Drive space (~200 GB for all three datasets processed)
- Google Colab (CPU or T4 GPU runtime)
- Kaggle account + API token (`~/.kaggle/kaggle.json`)
- HuggingFace account + access token

### Install dependencies

```bash
pip install -r requirements.txt
```

On Colab, `torch` and `torchvision` are pre-installed. You only need:

```bash
pip install -q datasets huggingface-hub kagglehub seaborn
```

---

## How to run

All commands are run from the project root (`licenta/`). On Colab, prefix with `!` or use a `%%bash` cell. Every script mounts Google Drive automatically on first run.

### 1. Download and prepare datasets

**Dataset A — Kaggle**

```bash
python scripts/download_dataset_a.py
python scripts/prepare_dataset_a.py
```

**Dataset B — HuggingFace**

```bash
python scripts/download_dataset_b.py
python scripts/prepare_dataset_b.py
```

Set your HuggingFace token via the `HF_TOKEN` environment variable before running.

**Dataset C — Kaggle**

```bash
python scripts/download_dataset_c.py
```

No prepare script needed — download handles everything in one pass.

All scripts are resumable — already-saved images are skipped on restart.

---

### 2. Train

```bash
python -m src.training.train_resnet50 --dataset ai_vs_human
python -m src.training.train_resnet50 --dataset dataset_b
python -m src.training.train_resnet50 --dataset dataset_c

python -m src.training.train_resnet18 --dataset ai_vs_human
python -m src.training.train_resnet18 --dataset dataset_b
python -m src.training.train_resnet18 --dataset dataset_c
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 5 | Number of training epochs |
| `--batch_size N` | 128 | Batch size |
| `--not_resized` | off | Pass if images are NOT pre-resized to 224 on disk |
| `--resume` | off | Resume from the latest checkpoint in the checkpoint directory |

A checkpoint is saved after every epoch to `checkpoints/<dataset>/<model>/`. The best validation accuracy is tracked and printed at the end of training.

---

### 3. Evaluate a single model

```bash
python -m src.evaluation.evaluate_resnet50 --dataset ai_vs_human
python -m src.evaluation.evaluate_resnet18 --dataset dataset_b
```

The script automatically finds the checkpoint with the highest validation accuracy. Outputs (metrics CSV + confusion matrix PNG) are saved to `logs/<dataset>/<model>/`.

---

### 4. Cross-dataset generalisation matrix

```bash
python -m src.evaluation.cross_dataset_matrix --model resnet50
python -m src.evaluation.cross_dataset_matrix --model resnet18
```

Iterates every `(trained_on × tested_on)` combination across all datasets in `config.py`. Missing checkpoints or datasets are skipped gracefully and shown as N/A.

Outputs are saved to `logs/cross_dataset_matrix/<model>/<NxN>/` where N is the number of datasets — e.g. `3x3/` for three datasets. This means results from different runs with different numbers of datasets are never overwritten.

---

## Configuration

All paths and hyperparameters are in `src/config.py`:

```python
DRIVE_ROOT  = "/content/drive/MyDrive/licenta"  # change if your Drive path differs
BATCH_SIZE  = 128
EPOCHS      = 5
```

To add a new dataset, add one entry to `DATASETS` — everything else (training, evaluation, matrix) picks it up automatically.

---

## Planned work

- ViT-B/16 experiments across all datasets
- Fourth dataset based on DiT-generated images
- Analysis of whether detectors capture deep generative artefacts or dataset-specific shortcuts across GAN, diffusion (U-Net), and diffusion (DiT) generation paradigms