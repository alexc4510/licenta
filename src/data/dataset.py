# src/data/dataset.py

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _safe_num_workers() -> int:
    """
    Cap at 2 workers for Colab.  Drive I/O + more workers = frequent
    DataLoader crashes.  Falls back to 0 if cpu_count is unavailable.
    """
    return min(2, os.cpu_count() or 1)


def _build_transforms(already_resized: bool = True):
    """
    Return (train_transform, eval_transform).

    already_resized=True  → images are exactly IMAGE_SIZExIMAGE_SIZE on disk
                            (saved by download_dataset_b / prepare_dataset_a).
                            Skip Resize/CenterCrop — saves meaningful time.
    already_resized=False → apply the standard Resize(256) → CenterCrop(224).
    """
    normalise = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    spatial = [] if already_resized else [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
    ]
    train_tf = transforms.Compose(spatial + [transforms.RandomHorizontalFlip()] + normalise)
    eval_tf  = transforms.Compose(spatial + normalise)
    return train_tf, eval_tf


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    already_resized: bool = True,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from an ImageFolder layout:

        data_dir/
            train/  ai/  real/
            val/    ai/  real/
            test/   ai/  real/

    ImageFolder assigns labels alphabetically → ai=0, real=1.
    This is consistent across all our datasets so no remapping is needed.
    """
    train_tf, eval_tf = _build_transforms(already_resized)
    nw = num_workers if num_workers is not None else _safe_num_workers()

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tf)

    # Sanity check: label mapping must be identical across splits
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, (
        f"class_to_idx mismatch between splits in {data_dir}.\n"
        f"  train={train_ds.class_to_idx}  val={val_ds.class_to_idx}  test={test_ds.class_to_idx}"
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(
        f"Loaded: {data_dir}\n"
        f"  train={len(train_ds):>6}  val={len(val_ds):>6}  test={len(test_ds):>6}"
        f"  |  class_to_idx={train_ds.class_to_idx}  num_workers={nw}"
    )
    return train_loader, val_loader, test_loader


def get_test_loader(
    data_dir: str,
    batch_size: int = 32,
    already_resized: bool = True,
    num_workers: int | None = None,
) -> DataLoader:
    """
    Convenience function that returns only the test DataLoader.
    Used by the cross-dataset matrix script to avoid loading train/val.
    """
    _, eval_tf = _build_transforms(already_resized)
    nw = num_workers if num_workers is not None else _safe_num_workers()
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)
    print(f"  Test set: {data_dir}/test  ({len(test_ds)} images)")
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )