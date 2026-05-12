# src/evaluation/evaluate_resnet18.py
"""
Evaluate the best ResNet-18 checkpoint on a single dataset's test split.

Checkpoint resolution priority:
    1. --experiment_name  — use this when --experiment_name was passed during training
    2. --checkpoint_dataset — use this when training used the default dataset name
    3. --dataset — fallback, assumes checkpoint and test dataset are the same

Flag consistency:
    --dog must match the flag used during training. A model trained with --dog
    expects DoG-preprocessed inputs at inference time. Mixing --dog between
    training and evaluation will produce meaningless results.

Output files are saved to:
    logs/<experiment_name or checkpoint_dataset>/<model>/
        <model>_<dataset>_test_metrics.csv
        <model>_<dataset>_confusion_matrix.png

Usage (from project root):
    # Standard evaluation (no DoG)
    python -m src.evaluation.evaluate_resnet18 --dataset dataset_b
    python -m src.evaluation.evaluate_resnet18 --dataset dataset_b --checkpoint_dataset dataset_combined

    # DoG evaluation (experiments 11/12)
    python -m src.evaluation.evaluate_resnet18 --dataset dataset_b --experiment_name dataset_combined_dog --dog
    python -m src.evaluation.evaluate_resnet18 --dataset dataset_c --experiment_name dataset_combined_dog --dog
    python -m src.evaluation.evaluate_resnet18 --dataset ai_vs_human --experiment_name dataset_combined_dog --dog

    # Balanced test split (experiments 7/8)
    python -m src.evaluation.evaluate_resnet18 --dataset dataset_b_balanced --test_split test_balanced
"""

from __future__ import annotations

import argparse
import os

import torch
from google.colab import drive

from src.config import BATCH_SIZE, CHECKPOINTS_ROOT, DATASETS, LOGS_ROOT, NUM_CLASSES
from src.data.dataset import get_test_loader
from src.evaluation._common import run_inference, save_single_eval
from src.models.resnet18 import get_resnet18
from src.training._common import find_best_checkpoint

MODEL_NAME = "resnet18"


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",            default="dataset_b", choices=list(DATASETS),
                   help="Dataset to evaluate on (determines test image path)")
    p.add_argument("--checkpoint_dataset", default=None, choices=list(DATASETS),
                   help="Dataset the model was trained on (determines checkpoint path). "
                        "Defaults to --dataset if not specified.")
    p.add_argument("--experiment_name",    default=None,
                   help="Override checkpoint/log path name. Use when --experiment_name "
                        "was passed during training (e.g. dataset_combined_dog).")
    p.add_argument("--test_split",         default="test",
                   help="Name of the test subfolder to use. Default: 'test'.")
    p.add_argument("--batch_size",         type=int, default=BATCH_SIZE)
    p.add_argument("--not_resized",        action="store_true")
    p.add_argument("--dog",                action="store_true",
                   help="Apply Difference of Gaussians preprocessing. "
                        "Must match the flag used during training.")
    p.add_argument("--save_metric",        default="val_acc", choices=["val_acc", "val_f1"],
                   help="Metric used to find best checkpoint (default: val_acc). "
                        "Use val_f1 for experiments 13/14 which saved best on val_f1.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _mount_drive()

    ckpt_dataset = args.experiment_name or args.checkpoint_dataset or args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device}  |  model={MODEL_NAME}")
    print(f"  checkpoint path:   {ckpt_dataset}")
    print(f"  evaluating on:     {args.dataset}/{args.test_split}")
    print(f"  dog preprocessing: {args.dog}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    test_loader = get_test_loader(
        DATASETS[args.dataset],
        batch_size=args.batch_size,
        already_resized=not args.not_resized,
        test_split=args.test_split,
        dog=args.dog,
    )

    model     = get_resnet18(num_classes=NUM_CLASSES).to(device)
    ckpt_dir  = os.path.join(CHECKPOINTS_ROOT, ckpt_dataset, MODEL_NAME)
    best_ckpt = find_best_checkpoint(ckpt_dir, MODEL_NAME, metric=args.save_metric)
    try:
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}  val_f1={ckpt.get('val_f1', float('nan')):.4f}")

    preds, labels = run_inference(model, test_loader, device)
    metrics_dir   = os.path.join(LOGS_ROOT, ckpt_dataset, MODEL_NAME)
    idx_to_class  = {v: k for k, v in test_loader.dataset.class_to_idx.items()}
    class_names   = [idx_to_class[i] for i in range(len(idx_to_class))]

    label = args.dataset if args.test_split == "test" else f"{args.dataset}_{args.test_split}"
    save_single_eval(preds, labels, metrics_dir, MODEL_NAME, label, class_names=class_names)


if __name__ == "__main__":
    main()