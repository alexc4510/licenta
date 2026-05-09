# src/training/train_resnet50.py
"""
Train ResNet-50 on a chosen dataset.

Checkpoints and logs are saved under:
    checkpoints/<experiment_name>/<model>/   resnet50_epoch_NN.pth
    logs/<experiment_name>/<model>/          training_metrics.csv  loss_curve.png  acc_curve.png

where <experiment_name> defaults to <dataset> if --experiment_name is not provided.

Flags:
    --dog              Apply Difference of Gaussians preprocessing (sigma_weak=1.0,
                       sigma_strong=2.0, kernel sizes 7x7 and 13x13). When used,
                       evaluation must also pass --dog to ensure consistent input
                       distribution between training and inference.
    --experiment_name  Override the directory name used for saving checkpoints and
                       logs. Use this when training on the same dataset as a previous
                       experiment to avoid overwriting results.
                       Example: --experiment_name dataset_combined_dog
    --resume           Resume from the latest (highest epoch number) checkpoint in
                       the checkpoint directory. Useful for continuing interrupted runs.

Usage (from project root in Colab):
    # Standard training
    python -m src.training.train_resnet50 --dataset dataset_b
    python -m src.training.train_resnet50 --dataset dataset_combined

    # DoG preprocessing experiment (experiments 11/12)
    python -m src.training.train_resnet50 --dataset dataset_combined --dog --experiment_name dataset_combined_dog

    # Resume an interrupted run
    python -m src.training.train_resnet50 --dataset dataset_b --resume
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
from src.models.resnet50 import get_resnet50
from src.training._common import (
    compute_class_weights, eval_epoch, load_latest_checkpoint,
    save_checkpoint, save_training_artifacts, train_epoch,
)

MODEL_NAME = "resnet50"


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    p.add_argument("--dataset",          default="dataset_b", choices=list(DATASETS))
    p.add_argument("--epochs",           type=int, default=EPOCHS)
    p.add_argument("--batch_size",       type=int, default=BATCH_SIZE)
    p.add_argument(
        "--not_resized", action="store_true",
        help="Images on disk are NOT pre-resized to 224 — apply Resize+CenterCrop at load time",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in the checkpoint directory",
    )
    p.add_argument(
        "--dog", action="store_true",
        help="Apply Difference of Gaussians preprocessing (sigma_weak=1.0, sigma_strong=2.0)",
    )
    p.add_argument(
        "--experiment_name", default=None,
        help="Override the dataset name used for checkpoint and log paths. "
             "Use this to avoid overwriting results from a previous experiment "
             "that used the same dataset. E.g. --experiment_name dataset_combined_dog",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _mount_drive()

    # experiment_name controls where checkpoints and logs are saved.
    # Falls back to dataset name if not specified.
    exp_name = args.experiment_name or args.dataset

    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, exp_name, MODEL_NAME)
    logs_dir       = os.path.join(LOGS_ROOT,        exp_name, MODEL_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir,       exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device}  |  model={MODEL_NAME}  |  dataset={args.dataset}")
    print(f"  experiment_name={exp_name}  |  dog={args.dog}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _ = get_dataloaders(
        DATASETS[args.dataset],
        batch_size=args.batch_size,
        already_resized=not args.not_resized,
        dog=args.dog,
    )

    model     = get_resnet50(num_classes=NUM_CLASSES).to(device)
    weights   = compute_class_weights(train_loader, NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 1
    if args.resume:
        start_epoch = load_latest_checkpoint(checkpoint_dir, MODEL_NAME, model, optimizer, device)

    history        = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc   = -1.0
    best_ckpt_path = ""

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 55}")
        print(f"  Epoch {epoch}/{args.epochs}  [{MODEL_NAME} | {exp_name}]")
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