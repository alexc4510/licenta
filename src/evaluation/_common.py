# src/evaluation/_common.py
"""
Pure inference and metrics logic.
No Drive mounting, no argparse — just functions.
Called by evaluate_resnet*.py and cross_dataset_matrix.py.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run *model* over every batch in *loader*.
    Returns (predictions, ground_truth) as numpy arrays.
    """
    model.eval()
    all_preds  = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Return accuracy, precision, recall and f1.

    Uses average="macro" — computes each metric per class then averages.
    This gives a balanced view regardless of class imbalance and is not
    affected by which class happens to be label 0 or label 1.
    """
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(   labels, preds, average="macro", zero_division=0),
        "f1":        f1_score(       labels, preds, average="macro", zero_division=0),
    }


def print_metrics(
    metrics: dict[str, float],
    model_name: str,
    train_dataset: str,
    test_dataset: str,
) -> None:
    header = f"[{model_name}  |  trained={train_dataset}  |  tested={test_dataset}]"
    print(f"\n{header}")
    print("─" * len(header))
    for k, v in metrics.items():
        print(f"  {k:<12}{v:.4f}")


def save_single_eval(
    preds: np.ndarray,
    labels: np.ndarray,
    metrics_dir: str,
    model_name: str,
    dataset_name: str,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute metrics, save CSV + confusion-matrix PNG + classification report
    to *metrics_dir*.  Returns the metrics dict.
    """
    os.makedirs(metrics_dir, exist_ok=True)

    # class_names follows ImageFolder alphabetical order: ai=0, real=1
    ticks = class_names or ["ai", "real"]

    m = compute_metrics(preds, labels)
    print_metrics(m, model_name, dataset_name, dataset_name)

    # Per-class breakdown — useful for diagnosing imbalance issues
    print(f"\n  Per-class report:")
    print(classification_report(labels, preds, target_names=ticks, zero_division=0))

    # CSV
    csv_path = os.path.join(metrics_dir, f"{model_name}_test_metrics.csv")
    pd.DataFrame({k: [v] for k, v in m.items()}).to_csv(csv_path, index=False)
    print(f"  Metrics → {csv_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ticks, yticklabels=ticks, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — {dataset_name}")
    cm_path = os.path.join(metrics_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix → {cm_path}")

    return m