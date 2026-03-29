# src/training/_common.py
"""
Shared training/evaluation loop logic.
Imported by train_resnet18.py and train_resnet50.py — never run directly.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive; safe for scripts and Colab cells
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    train_loader: DataLoader,
    num_classes: int,
    device: str,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training set.

    Formula:  weight[i] = total_samples / (num_classes * count[i])

    For a balanced dataset all weights come out to 1.0 — no effect on loss.
    For an imbalanced dataset minority classes get higher weights, penalising
    the model more for misclassifying them.

    Works with any ImageFolder-based DataLoader via .dataset.targets.
    Prints a per-class breakdown so the values are always visible in the log.
    """
    targets      = torch.tensor(train_loader.dataset.targets)
    total        = len(targets)
    weights      = torch.zeros(num_classes)
    idx_to_class = {v: k for k, v in train_loader.dataset.class_to_idx.items()}

    for c in range(num_classes):
        count      = (targets == c).sum().item()
        weights[c] = total / (num_classes * count) if count > 0 else 1.0
        print(f"  class '{idx_to_class[c]}' (idx={c}): count={count}  weight={weights[c]:.4f}")

    return weights.to(device)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_every: int = 50,
) -> tuple[float, float]:
    """One full training pass.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = total = 0
    start = time.time()

    for i, (images, labels) in enumerate(loader):
        t_batch = time.time()
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if i % log_every == 0:
            done  = i + 1
            eta   = (len(loader) - done) * (time.time() - start) / done
            print(
                f"  Batch {done:>4}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"batch={time.time()-t_batch:.2f}s | "
                f"ETA={eta/60:.1f}min"
            )

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """One full evaluation pass.  Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    checkpoint_dir: str,
    model_name: str,
) -> str:
    """
    Save a full checkpoint that includes val_acc and epoch metadata.
    Returns the saved path.

    File format:  <model_name>_epoch_<NN>.pth
    Stored dict:
        epoch                int
        val_acc              float
        model_state_dict     OrderedDict
        optimizer_state_dict OrderedDict
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch:02d}.pth")
    torch.save(
        {
            "epoch":                epoch,
            "val_acc":              val_acc,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return path


def find_best_checkpoint(checkpoint_dir: str, model_name: str) -> str:
    """
    Scan *checkpoint_dir* for files matching <model_name>_epoch_*.pth,
    read the val_acc stored inside each, and return the path of the
    checkpoint with the highest validation accuracy.

    Raises FileNotFoundError if no valid checkpoints are found.
    """
    candidates = sorted(Path(checkpoint_dir).glob(f"{model_name}_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matching '{model_name}_epoch_*.pth' in:\n  {checkpoint_dir}"
        )

    best_path: Path | None = None
    best_acc = -1.0

    for p in candidates:
        try:
            ckpt    = torch.load(p, map_location="cpu", weights_only=True)
            val_acc = float(ckpt.get("val_acc", -1.0))
            if val_acc > best_acc:
                best_acc  = val_acc
                best_path = p
        except Exception as exc:
            print(f"  [warn] Skipping unreadable checkpoint {p.name}: {exc}")

    if best_path is None:
        raise FileNotFoundError(
            f"Found checkpoint files but none could be read in:\n  {checkpoint_dir}"
        )

    print(f"  Best checkpoint: {best_path.name}  (val_acc={best_acc:.4f})")
    return str(best_path)


def load_latest_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    """
    Load the most recently saved checkpoint (highest epoch number) from
    *checkpoint_dir* into *model* and *optimizer*.
    Returns the epoch number to resume from (last completed epoch + 1).
    Returns 1 if no checkpoint is found (fresh start).
    """
    candidates = sorted(Path(checkpoint_dir).glob(f"{model_name}_epoch_*.pth"))
    if not candidates:
        print("  No checkpoint found — starting from scratch.")
        return 1

    latest = candidates[-1]
    try:
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = int(ckpt["epoch"])
        print(f"  Resumed from: {latest.name}  (epoch={epoch}  val_acc={ckpt.get('val_acc', float('nan')):.4f})")
        return epoch + 1
    except Exception as exc:
        print(f"  [warn] Could not load checkpoint {latest.name}: {exc} — starting from scratch.")
        return 1

def save_training_artifacts(
    history: dict[str, list[Any]],
    logs_dir: str,
    model_name: str,
) -> None:
    """Save training history CSV and loss/accuracy curve PNGs to *logs_dir*."""
    os.makedirs(logs_dir, exist_ok=True)
    df = pd.DataFrame(history)

    csv_path = os.path.join(logs_dir, f"{model_name}_training_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved:  {csv_path}")

    for metric, title in [("loss", "Loss"), ("acc", "Accuracy")]:
        fig, ax = plt.subplots()
        ax.plot(df["epoch"], df[f"train_{metric}"], label="Train")
        ax.plot(df["epoch"], df[f"val_{metric}"],   label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{model_name} — {title}")
        ax.legend()
        path = os.path.join(logs_dir, f"{model_name}_{metric}_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved:     {path}")