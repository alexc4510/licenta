# scripts/build_dataset_b_metadata.py
"""
Builds a metadata CSV for dataset_b by re-loading the HuggingFace dataset
(metadata only — no image re-download) and mapping each item to its saved
filename on Drive.

The CSV is saved to:
    datasets/dataset_b/metadata.csv

Columns:
    filename   — e.g. train_000000.png  (matches the file saved by download_dataset_b.py)
    split      — train / val / test
    label_a    — 0=real, 1=ai  (binary classification label)
    label_b    — 0=real, 1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E3, 5=MidJourney v6
    caption    — original MS-COCO caption

Run once after download_dataset_b.py has completed.
Re-running is safe — overwrites the existing CSV.

Usage:
    python scripts/build_dataset_b_metadata.py
"""

import os
import csv

from google.colab import drive
from datasets import load_dataset
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

DRIVE_ROOT   = "/content/drive/MyDrive/licenta"
DATASET_DIR  = os.path.join(DRIVE_ROOT, "datasets", "dataset_b")
METADATA_CSV = os.path.join(DATASET_DIR, "metadata.csv")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# HuggingFace split name → our folder name (must match download_dataset_b.py)
SPLIT_FOLDER = {
    "train":      "train",
    "validation": "val",
    "test":       "test",
}

# Label_B mapping for reference
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

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _mount_drive()

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}\n"
            "Run download_dataset_b.py first."
        )

    print("\nLoading dataset metadata from HuggingFace Hub …")
    print("(This loads metadata only — images are not re-downloaded)")
    from huggingface_hub import login
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)

    dataset = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        trust_remote_code=True,
    )

    records = []
    total   = 0

    for hf_split, folder_name in SPLIT_FOLDER.items():
        data = dataset[hf_split]
        n    = len(data)
        print(f"\n  Processing split: {hf_split} ({n} items) …")

        for i in tqdm(range(n), desc=hf_split, unit="item"):
            try:
                item = data[i]
            except Exception as exc:
                print(f"\n  [!] Cannot read item {i}: {exc}")
                continue

            label_a  = item.get("Label_A", -1)
            label_b  = item.get("Label_B", -1)
            caption  = item.get("Caption", "")
            filename = f"{hf_split}_{i:06d}.png"

            records.append({
                "filename": filename,
                "split":    folder_name,
                "label_a":  label_a,
                "label_b":  label_b,
                "caption":  caption,
            })
            total += 1

    print(f"\nWriting metadata CSV → {METADATA_CSV}")
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "split", "label_a", "label_b", "caption"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"  Wrote {total} rows.")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\nLabel_B distribution per split:")
    from collections import Counter
    by_split: dict[str, Counter] = {}
    for r in records:
        s = r["split"]
        by_split.setdefault(s, Counter())
        by_split[s][r["label_b"]] += 1

    for split, counts in by_split.items():
        print(f"\n  {split}:")
        for lb in sorted(counts):
            name = LABEL_B_MAP.get(lb, f"unknown_{lb}")
            print(f"    label_b={lb} ({name:<16}): {counts[lb]:>6}")

    print(f"\n{'═'*60}")
    print(f"Metadata saved to: {METADATA_CSV}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()