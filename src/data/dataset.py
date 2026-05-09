# src/data/dataset.py

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _safe_num_workers() -> int:
    """
    Cap at 2 workers for Colab.  Drive I/O + more workers = frequent
    DataLoader crashes.  Falls back to 0 if cpu_count is unavailable.
    """
    return min(2, os.cpu_count() or 1)


class DifferenceOfGaussians:
    """
    Difference of Gaussians (DoG) transform.

    Applies two Gaussian blurs with different standard deviations to the
    input image and returns their difference. The result highlights edges
    and high-frequency artefacts while suppressing low-frequency content.

    This is applied AFTER ToTensor() so the input is a float tensor in [0, 1].
    The output is clamped back to [0, 1] before normalization.

    Parameters
    ----------
    sigma_weak  : float
        Standard deviation of the weak (less blurry) Gaussian.
        Default: 1.0  →  kernel 7x7  (covers 3σ in each direction)
    sigma_strong : float
        Standard deviation of the strong (more blurry) Gaussian.
        Default: 2.0  →  kernel 13x13

    Kernel size formula (standard in literature):
        kernel_size = 2 * ceil(3 * sigma) + 1
    """

    def __init__(self, sigma_weak: float = 1.0, sigma_strong: float = 2.0):
        import math
        self.sigma_weak   = sigma_weak
        self.sigma_strong = sigma_strong

        def _kernel(sigma: float) -> int:
            return 2 * math.ceil(3 * sigma) + 1

        self.blur_weak   = transforms.GaussianBlur(kernel_size=_kernel(sigma_weak),   sigma=sigma_weak)
        self.blur_strong = transforms.GaussianBlur(kernel_size=_kernel(sigma_strong), sigma=sigma_strong)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        weak   = self.blur_weak(tensor)
        strong = self.blur_strong(tensor)
        dog    = weak - strong
        # Shift and scale to [0, 1] so ImageNet normalization remains meaningful
        dog = dog - dog.min()
        max_val = dog.max()
        if max_val > 0:
            dog = dog / max_val
        return dog

    def __repr__(self) -> str:
        return (f"DifferenceOfGaussians("
                f"sigma_weak={self.sigma_weak}, sigma_strong={self.sigma_strong})")


def _build_transforms(already_resized: bool = True, dog: bool = False):
    """
    Return (train_transform, eval_transform).

    already_resized=True  → images are exactly IMAGE_SIZExIMAGE_SIZE on disk.
                            Skip Resize/CenterCrop — saves meaningful time.
    already_resized=False → apply the standard Resize(256) → CenterCrop(224).

    dog=True  → apply Difference of Gaussians after ToTensor and before
                normalization. Extracts edge/contour information.
    dog=False → standard pipeline, no DoG preprocessing.
    """
    spatial = [] if already_resized else [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
    ]

    to_tensor = [transforms.ToTensor()]

    dog_transform = [DifferenceOfGaussians(sigma_weak=1.0, sigma_strong=2.0)] if dog else []

    normalise = [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    train_tf = transforms.Compose(
        spatial + [transforms.RandomHorizontalFlip()] + to_tensor + dog_transform + normalise
    )
    eval_tf = transforms.Compose(
        spatial + to_tensor + dog_transform + normalise
    )
    return train_tf, eval_tf


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    already_resized: bool = True,
    num_workers: int | None = None,
    dog: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from an ImageFolder layout:

        data_dir/
            train/  ai/  real/
            val/    ai/  real/
            test/   ai/  real/

    ImageFolder assigns labels alphabetically → ai=0, real=1.
    This is consistent across all our datasets so no remapping is needed.

    dog=True applies Difference of Gaussians preprocessing to all splits.
    """
    train_tf, eval_tf = _build_transforms(already_resized, dog=dog)
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

    dog_str = "  [DoG preprocessing: ON]" if dog else ""
    print(
        f"Loaded: {data_dir}\n"
        f"  train={len(train_ds):>6}  val={len(val_ds):>6}  test={len(test_ds):>6}"
        f"  |  class_to_idx={train_ds.class_to_idx}  num_workers={nw}{dog_str}"
    )
    return train_loader, val_loader, test_loader


def get_test_loader(
    data_dir: str,
    batch_size: int = 32,
    already_resized: bool = True,
    num_workers: int | None = None,
    test_split: str = "test",
    dog: bool = False,
) -> DataLoader:
    """
    Convenience function that returns only the test DataLoader.
    Used by the cross-dataset matrix script to avoid loading train/val.

    test_split — name of the subfolder to use as the test set.
                 Defaults to "test". Pass "test_balanced" for the
                 balanced test split of dataset_b_balanced.

    dog=True applies Difference of Gaussians preprocessing.
    """
    _, eval_tf = _build_transforms(already_resized, dog=dog)
    nw = num_workers if num_workers is not None else _safe_num_workers()
    test_ds = datasets.ImageFolder(os.path.join(data_dir, test_split), transform=eval_tf)
    dog_str = "  [DoG: ON]" if dog else ""
    print(f"  Test set: {data_dir}/{test_split}  ({len(test_ds)} images){dog_str}")
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )