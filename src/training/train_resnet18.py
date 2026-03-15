# src/training/train_resnet18.py
"""
Train ResNet-18 on a chosen dataset.

Usage (from project root in Colab):
    python -m src.training.train_resnet18 --dataset dataset_b
    python -m src.training.train_resnet18 --dataset ai_vs_human
    python -m src.training.train_resnet18 --dataset dataset_b --epochs 5 --batch_size 64
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from google.colab import drive

from src.config import BATCH_SIZE, CHECKPOINTS_ROOT, DATASETS, EPOCHS, LOGS_ROOT, NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.models.resnet18 import get_resnet18
from src.training._common import (
    eval_epoch, find_best_checkpoint, save_checkpoint,
    save_training_artifacts, train_epoch,
)

MODEL_NAME = "resnet18"


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    p.add_argument("--dataset",     default="dataset_b", choices=list(DATASETS))
    p.add_argument("--epochs",      type=int, default=EPOCHS)
    p.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    p.add_argument(
        "--not_resized", action="store_true",
        help="Images on disk are NOT pre-resized to 224 — apply Resize+CenterCrop at load time",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _mount_drive()

    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, args.dataset, MODEL_NAME)
    logs_dir       = os.path.join(LOGS_ROOT,        args.dataset, MODEL_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir,       exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device}  |  model={MODEL_NAME}  |  dataset={args.dataset}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _ = get_dataloaders(
        DATASETS[args.dataset],
        batch_size=args.batch_size,
        already_resized=not args.not_resized,
    )

    model     = get_resnet18(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history        = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc   = -1.0
    best_ckpt_path = ""

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 55}")
        print(f"  Epoch {epoch}/{args.epochs}  [{MODEL_NAME} | {args.dataset}]")
        print(f"{'=' * 55}")
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, device)

        print(
            f"\n  Summary | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"time={(time.time()-t0)/60:.1f}min"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        ckpt = save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, MODEL_NAME)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_ckpt_path = ckpt
            print(f"  ★ New best  val_acc={best_val_acc:.4f}  →  {ckpt}")

    print(f"\nTraining complete.  Best checkpoint: {best_ckpt_path}")
    save_training_artifacts(history, logs_dir, MODEL_NAME)


if __name__ == "__main__":
    main()