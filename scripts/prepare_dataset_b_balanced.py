# scripts/prepare_dataset_b_balanced.py
"""
Creates a balanced version of dataset_b (dataset_b_balanced) by subsampling
AI images equally across all five generator subcategories, using the metadata
CSV produced by build_dataset_b_metadata.py.

Balancing logic:
    Train:  7,000 real  +  7,000 AI  (1,400 per generator x 5 generators)
    Val:    1,500 real  +  1,500 AI  (  300 per generator x 5 generators)
    Test (original):  7,500 real + 37,500 AI  — unchanged, reflects real-world distribution
    Test (balanced):  7,500 real  +  7,500 AI  (1,500 per generator x 5 generators)

Images are copied (not symlinked) from datasets/dataset_b/ to
datasets/dataset_b_balanced/ to keep the folder self-contained and
compatible with ImageFolder without any changes to dataset.py.

Run once. Re-running is safe — already-copied images are skipped.

Usage:
    python scripts/prepare_dataset_b_balanced.py
"""

from __future__ import annotations

import os
import random
import shutil
import csv
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from google.colab import drive

# ── CONFIG ────────────────────────────────────────────────────────────────────

DRIVE_ROOT   = "/content/drive/MyDrive/licenta"
SRC_DIR      = os.path.join(DRIVE_ROOT, "datasets", "dataset_b")
DST_DIR      = os.path.join(DRIVE_ROOT, "datasets", "dataset_b_balanced")
METADATA_CSV = os.path.join(SRC_DIR, "metadata.csv")

RANDOM_SEED  = 42

# How many images to keep per class per split
# AI images are further split equally across generators (label_b 1-5)
TARGETS = {
    "train": {"real": 7000,  "ai_per_generator": 1400},   # 7k real + 7k AI
    "val":   {"real": 1500,  "ai_per_generator":  300},   # 1.5k real + 1.5k AI
    "test_original": None,                                  # copy all — unchanged
    "test_balanced": {"real": 7500, "ai_per_generator": 1500},  # 7.5k real + 7.5k AI
}

GENERATORS = [1, 2, 3, 4, 5]  # label_b values for AI images

LABEL_B_MAP = {
    0: "real",
    1: "SD2.1",
    2: "SDXL",
    3: "SD3",
    4: "DALL-E3",
    5: "MidJourney_v6",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Mounting Google Drive …")
        drive.mount("/content/drive")
    else:
        print("Google Drive already mounted.")


def _make_dirs() -> None:
    for split in ["train", "val", "test", "test_balanced"]:
        for label in ["ai", "real"]:
            os.makedirs(os.path.join(DST_DIR, split, label), exist_ok=True)


def _load_metadata() -> list[dict]:
    records = []
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label_a"] = int(row["label_a"])
            row["label_b"] = int(row["label_b"])
            records.append(row)
    return records


def _copy_file(src: str, dst: str) -> bool:
    if os.path.exists(dst):
        return False  # already copied — skip
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        print(f"\n  [!] Failed to copy {src}: {exc}")
        return False


def _process_split(
    records: list[dict],
    hf_split: str,
    dst_split: str,
    real_target: int,
    ai_per_generator: int,
) -> None:
    """Select and copy the balanced subset for one split."""

    split_records = [r for r in records if r["split"] == hf_split]

    # Separate real and AI by generator
    real_records = [r for r in split_records if r["label_a"] == 0]
    ai_by_gen: dict[int, list[dict]] = defaultdict(list)
    for r in split_records:
        if r["label_a"] == 1:
            ai_by_gen[r["label_b"]].append(r)

    # Sample real images
    random.seed(RANDOM_SEED)
    selected_real = random.sample(real_records, min(real_target, len(real_records)))

    # Sample AI images equally per generator
    selected_ai = []
    for gen in GENERATORS:
        pool = ai_by_gen.get(gen, [])
        n    = min(ai_per_generator, len(pool))
        if n < ai_per_generator:
            print(f"  [warn] Generator {gen} ({LABEL_B_MAP[gen]}): only {n} available, needed {ai_per_generator}")
        selected_ai.extend(random.sample(pool, n))

    print(f"\n{'─'*60}")
    print(f"  Split: {hf_split} → {dst_split}")
    print(f"  real selected:  {len(selected_real)}")
    print(f"  AI selected:    {len(selected_ai)} ({len(selected_ai)//len(GENERATORS)} per generator)")
    print(f"{'─'*60}")

    copied = skipped = 0

    for r in tqdm(selected_real, desc=f"{dst_split}/real", unit="img"):
        src = os.path.join(SRC_DIR, r["split"], "real", r["filename"])
        dst = os.path.join(DST_DIR, dst_split, "real", r["filename"])
        if _copy_file(src, dst):
            copied += 1
        else:
            skipped += 1

    for r in tqdm(selected_ai, desc=f"{dst_split}/ai", unit="img"):
        src = os.path.join(SRC_DIR, r["split"], "ai", r["filename"])
        dst = os.path.join(DST_DIR, dst_split, "ai", r["filename"])
        if _copy_file(src, dst):
            copied += 1
        else:
            skipped += 1

    print(f"  ✓  copied={copied}  skipped={skipped}")


def _copy_full_split(
    records: list[dict],
    hf_split: str,
    dst_split: str,
) -> None:
    """Copy all images for a split without any subsampling."""
    split_records = [r for r in records if r["split"] == hf_split]

    print(f"\n{'─'*60}")
    print(f"  Split: {hf_split} → {dst_split}  (full copy, no subsampling)")
    print(f"  total images: {len(split_records)}")
    print(f"{'─'*60}")

    copied = skipped = 0

    for r in tqdm(split_records, desc=dst_split, unit="img"):
        label_folder = "real" if r["label_a"] == 0 else "ai"
        src = os.path.join(SRC_DIR, r["split"], label_folder, r["filename"])
        dst = os.path.join(DST_DIR, dst_split, label_folder, r["filename"])
        if _copy_file(src, dst):
            copied += 1
        else:
            skipped += 1

    print(f"  ✓  copied={copied}  skipped={skipped}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _mount_drive()

    if not os.path.isfile(METADATA_CSV):
        raise FileNotFoundError(
            f"Metadata CSV not found: {METADATA_CSV}\n"
            "Run build_dataset_b_metadata.py first."
        )

    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(
            f"Source dataset not found: {SRC_DIR}\n"
            "Run download_dataset_b.py first."
        )

    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    _make_dirs()

    print("\nLoading metadata …")
    records = _load_metadata()
    print(f"  Loaded {len(records)} records.")

    # ── Train (balanced) ──────────────────────────────────────────────────────
    _process_split(
        records,
        hf_split="train",
        dst_split="train",
        real_target=TARGETS["train"]["real"],
        ai_per_generator=TARGETS["train"]["ai_per_generator"],
    )

    # ── Val (balanced) ────────────────────────────────────────────────────────
    _process_split(
        records,
        hf_split="val",
        dst_split="val",
        real_target=TARGETS["val"]["real"],
        ai_per_generator=TARGETS["val"]["ai_per_generator"],
    )

    # ── Test (original — full copy, unchanged) ────────────────────────────────
    _copy_full_split(
        records,
        hf_split="test",
        dst_split="test",
    )

    # ── Test (balanced) ───────────────────────────────────────────────────────
    _process_split(
        records,
        hf_split="test",
        dst_split="test_balanced",
        real_target=TARGETS["test_balanced"]["real"],
        ai_per_generator=TARGETS["test_balanced"]["ai_per_generator"],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("DONE. Final counts:")
    for split in ["train", "val", "test", "test_balanced"]:
        for label in ["ai", "real"]:
            folder = os.path.join(DST_DIR, split, label)
            count  = len(list(Path(folder).glob("*.png"))) if os.path.isdir(folder) else 0
            print(f"  {split}/{label}: {count}")
    print(f"\nOutput: {DST_DIR}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()