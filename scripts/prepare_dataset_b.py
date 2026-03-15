"""
prepare_dataset_b.py

Verifies the processed dataset that was saved to Google Drive by
download_dataset_b.py, prints a per-split/label count table, and writes a
lightweight CSV manifest (path, split, label) so you can feed it directly
into a PyTorch Dataset without scanning the filesystem every time.

Run this after download_dataset_b.py completes (or after resuming a partial
download) to confirm everything looks healthy before training.
"""

import csv
import os
from pathlib import Path

from google.colab import drive

# ── CONFIG ────────────────────────────────────────────────────────────────────

DRIVE_ROOT   = "/content/drive/MyDrive/licenta"
DATASET_DIR  = os.path.join(DRIVE_ROOT, "datasets", "dataset_b")
MANIFEST_CSV = os.path.join(DATASET_DIR, "manifest.csv")

SPLITS = ["train", "val", "test"]
LABELS = ["real", "ai"]

# Expected counts (Label_A: 0=real, 1=ai).
# The full dataset has 96 000 images total.
# These are approximate — use them as a sanity check only.
EXPECTED = {
    "train": {"real": 7000, "ai": 35000},   # 42 000 total
    "val":   {"real": 1500, "ai":  7500},   #  9 000 total
    "test":  {"real": 7500, "ai": 37500},   # 45 000 total
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Mounting Google Drive …")
        drive.mount("/content/drive")
    else:
        print("Google Drive already mounted.")


def count_images(directory: str) -> int:
    """Count .png files directly inside *directory* (non-recursive)."""
    p = Path(directory)
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.suffix.lower() == ".png")


def build_manifest() -> list[dict]:
    """Walk the dataset directory and collect all image records."""
    records = []
    for split in SPLITS:
        for label in LABELS:
            folder = os.path.join(DATASET_DIR, split, label)
            p = Path(folder)
            if not p.is_dir():
                continue
            label_int = 0 if label == "real" else 1
            for img_path in sorted(p.glob("*.png")):
                records.append({
                    "path":  str(img_path),
                    "split": split,
                    "label": label_int,
                })
    return records

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mount_drive()

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}\n"
            "Run download_dataset_b.py first."
        )

    # ── Count table ───────────────────────────────────────────────────────────
    col_w = 12
    header = f"{'split':<{col_w}} {'label':<{col_w}} {'found':>{col_w}} {'expected':>{col_w}} {'status':>{col_w}}"
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    grand_total  = 0
    any_mismatch = False

    for split in SPLITS:
        for label in LABELS:
            folder    = os.path.join(DATASET_DIR, split, label)
            found     = count_images(folder)
            expected  = EXPECTED.get(split, {}).get(label, "?")
            if isinstance(expected, int):
                ok     = found >= expected * 0.97   # allow ≤3 % missing
                status = "✓ OK" if ok else "⚠ LOW"
                if not ok:
                    any_mismatch = True
            else:
                status = "?"
            print(f"{split:<{col_w}} {label:<{col_w}} {found:>{col_w}} {str(expected):>{col_w}} {status:>{col_w}}")
            grand_total += found

    print(f"{'─' * len(header)}")
    print(f"{'TOTAL':<{col_w}} {'':<{col_w}} {grand_total:>{col_w}}")
    print()

    if any_mismatch:
        print("⚠  Some splits appear incomplete. Re-run download_dataset_b.py to resume.\n")
    else:
        print("✓  All splits look complete.\n")

    # ── Manifest CSV ──────────────────────────────────────────────────────────
    print(f"Building manifest CSV → {MANIFEST_CSV}")
    records = build_manifest()

    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "split", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"  Wrote {len(records)} rows.")

    # ── Quick per-split breakdown ─────────────────────────────────────────────
    from collections import Counter
    counts: dict[str, Counter] = {}
    for r in records:
        counts.setdefault(r["split"], Counter())
        counts[r["split"]][r["label"]] += 1

    print("\nManifest breakdown:")
    for split in SPLITS:
        c = counts.get(split, Counter())
        total = sum(c.values())
        print(f"  {split:<6}  real={c[0]:>6}  ai={c[1]:>6}  total={total:>7}")

    print(f"\nManifest saved to: {MANIFEST_CSV}")
    print("Use it in your PyTorch Dataset like this:\n")
    print(
        "  import pandas as pd\n"
        "  df = pd.read_csv(MANIFEST_CSV)\n"
        "  train_df = df[df['split'] == 'train'].reset_index(drop=True)\n"
    )


if __name__ == "__main__":
    main()