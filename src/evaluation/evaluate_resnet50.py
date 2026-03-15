# src/evaluation/evaluate_resnet50.py
"""
Evaluate the best ResNet-50 checkpoint on a single dataset's test split.

Usage (from project root):
    python -m src.evaluation.evaluate_resnet50 --dataset dataset_b
    python -m src.evaluation.evaluate_resnet50 --dataset ai_vs_human
"""

from __future__ import annotations

import argparse
import os

import torch
from google.colab import drive

from src.config import BATCH_SIZE, CHECKPOINTS_ROOT, DATASETS, LOGS_ROOT, NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.evaluation._common import run_inference, save_single_eval
from src.models.resnet50 import get_resnet50
from src.training._common import find_best_checkpoint

MODEL_NAME = "resnet50"


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="dataset_b", choices=list(DATASETS))
    p.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    p.add_argument("--not_resized", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _mount_drive()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device}  |  model={MODEL_NAME}  |  dataset={args.dataset}")

    _, _, test_loader = get_dataloaders(
        DATASETS[args.dataset],
        batch_size=args.batch_size,
        already_resized=not args.not_resized,
    )

    model     = get_resnet50(num_classes=NUM_CLASSES).to(device)
    ckpt_dir  = os.path.join(CHECKPOINTS_ROOT, args.dataset, MODEL_NAME)
    best_ckpt = find_best_checkpoint(ckpt_dir, MODEL_NAME)
    ckpt      = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

    preds, labels = run_inference(model, test_loader, device)
    metrics_dir   = os.path.join(LOGS_ROOT, args.dataset, MODEL_NAME)
    save_single_eval(preds, labels, metrics_dir, MODEL_NAME, args.dataset)


if __name__ == "__main__":
    main()