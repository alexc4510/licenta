# scripts/prepare_dataset_a.py
"""
Prepares the ai-vs-human dataset for training:
  • Reads train.csv  →  stratified 80/10/10 split
  • Resizes every image to 224x224 (center-crop, same pipeline as dataset_b)
  • Saves to  datasets/ai_vs_human/{train,val,test}/{ai,real}/  on Google Drive

Resumable: already-saved images are skipped, so interrupted runs continue
safely from where they left off.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from google.colab import drive

DRIVE_ROOT = "/content/drive/MyDrive/licenta"
RAW_ROOT   = os.path.join(DRIVE_ROOT, "raw_data", "ai_vs_human")
DST_ROOT   = os.path.join(DRIVE_ROOT, "datasets", "ai_vs_human")

IMG_SIZE    = (224, 224)
MAX_RETRIES = 3
LABEL_FOLDER = {0: "real", 1: "ai"}


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _find_raw_base(root: str) -> str:
    """
    Walk *root* to find the directory containing train.csv.
    Handles kagglehub's  versions/N/  nesting transparently.
    """
    for dirpath, _, filenames in os.walk(root):
        if "train.csv" in filenames:
            return dirpath
    raise FileNotFoundError(
        f"train.csv not found anywhere under {root}.\n"
        "Run download_dataset_a.py first."
    )


def _preprocess(img: Image.Image) -> Image.Image:
    """RGB → scale shortest side to 224 → center-crop 224×224."""
    img   = img.convert("RGB")
    w, h  = img.size
    scale = IMG_SIZE[0] / min(w, h)
    nw    = max(IMG_SIZE[0], int(round(w * scale)))
    nh    = max(IMG_SIZE[1], int(round(h * scale)))
    img   = img.resize((nw, nh), Image.LANCZOS)
    left  = (nw - IMG_SIZE[0]) // 2
    upper = (nh - IMG_SIZE[1]) // 2
    return img.crop((left, upper, left + IMG_SIZE[0], upper + IMG_SIZE[1]))


def _save_with_retry(img: Image.Image, path: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            img.save(path, format="PNG", optimize=False)
            return True
        except OSError as exc:
            print(f"    Save attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(1.0 * attempt)
    return False


def main() -> None:
    _mount_drive()

    raw_base = _find_raw_base(RAW_ROOT)
    print(f"Raw data: {raw_base}")

    csv_path = os.path.join(raw_base, "train.csv")
    df       = pd.read_csv(csv_path)[["file_name", "label"]]
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts().rename("count").to_string())

    # Stratified 80 / 10 / 10
    train_df, temp_df = train_test_split(df,       test_size=0.20, stratify=df["label"],       random_state=42)
    val_df,   test_df = train_test_split(temp_df,  test_size=0.50, stratify=temp_df["label"],  random_state=42)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    print(f"\nSplit sizes: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Create all destination dirs upfront
    for split in splits:
        for lf in LABEL_FOLDER.values():
            os.makedirs(os.path.join(DST_ROOT, split, lf), exist_ok=True)

    total_saved = total_skipped = total_failed = 0

    for split, data in splits.items():
        print(f"\n{'─'*55}")
        print(f"Split: {split}  ({len(data)} images)")
        print(f"{'─'*55}")
        saved = skipped = failed = 0

        for row in tqdm(data.itertuples(index=False), total=len(data), desc=split, unit="img"):
            label_folder = LABEL_FOLDER.get(int(row.label))
            if label_folder is None:
                print(f"\n  [!] Unknown label={row.label!r} for {row.file_name}")
                failed += 1
                continue

            dst_path = str(
                Path(os.path.join(DST_ROOT, split, label_folder, Path(row.file_name).name))
                .with_suffix(".png")
            )

            if os.path.exists(dst_path):
                skipped += 1
                continue

            # Try several candidate source paths to handle kagglehub nesting
            src_candidates = [
                os.path.join(raw_base, row.file_name),
                os.path.join(raw_base, "train_data", Path(row.file_name).name),
                os.path.join(raw_base, Path(row.file_name).name),
            ]
            src_path = next((p for p in src_candidates if os.path.exists(p)), None)

            if src_path is None:
                print(f"\n  [!] Source file not found: {row.file_name}")
                failed += 1
                continue

            try:
                processed = _preprocess(Image.open(src_path))
            except (UnidentifiedImageError, Exception) as exc:
                print(f"\n  [!] Cannot open {row.file_name}: {exc}")
                failed += 1
                continue

            if _save_with_retry(processed, dst_path):
                saved += 1
            else:
                print(f"\n  [!] Permanently failed to save {row.file_name}")
                failed += 1

        print(f"  ✓  saved={saved}  skipped={skipped}  failed={failed}")
        total_saved   += saved
        total_skipped += skipped
        total_failed  += failed

    print(f"\n{'═'*55}")
    print(f"DONE  |  saved={total_saved}  skipped={total_skipped}  failed={total_failed}")
    print(f"Output: {DST_ROOT}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()