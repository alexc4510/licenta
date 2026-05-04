# scripts/prepare_dataset_combined.py
"""
Creates dataset_combined by merging subsets of Dataset B (diffusion-generated)
and Dataset C (GAN-generated) into a single balanced training and validation set.

Construction logic:
    Train:
        real — 7,000 from dataset_b/train/real  +  7,000 from dataset_c/train/real  =  14,000
        AI   — 7,000 from dataset_b/train/ai    +  7,000 from dataset_c/train/ai    =  14,000
               dataset_b AI: 1,400 per generator × 5 generators (uses metadata.csv)
               dataset_c AI: random sample of 7,000 from ~50,000 available
        Total: 28,000  |  balanced 1:1  |  AI: 50% diffusion + 50% GAN

    Val:
        real — 1,500 from dataset_b/val/real  +  1,500 from dataset_c/val/real  =  3,000
        AI   — 1,500 from dataset_b/val/ai    +  1,500 from dataset_c/val/ai    =  3,000
               dataset_b AI: 300 per generator × 5 generators
               dataset_c AI: random sample of 1,500 from ~10,000 available
        Total: 6,000  |  balanced 1:1

    Test:
        No test split — evaluation is done directly against the original test splits of
        dataset_b, dataset_c and ai_vs_human using the --checkpoint_dataset flag.

Prerequisites:
    - datasets/dataset_b/              (run download_dataset_b.py first)
    - datasets/dataset_b/metadata.csv  (run build_dataset_b_metadata.py first)
    - datasets/dataset_c/              (run download_dataset_c.py first)

Run once. Re-running is safe — already-copied images are skipped.

Usage:
    python scripts/prepare_dataset_combined.py
"""

from __future__ import annotations

import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from google.colab import drive
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

DRIVE_ROOT   = "/content/drive/MyDrive/licenta"
SRC_B        = os.path.join(DRIVE_ROOT, "datasets", "dataset_b")
SRC_C        = os.path.join(DRIVE_ROOT, "datasets", "dataset_c")
DST_DIR      = os.path.join(DRIVE_ROOT, "datasets", "dataset_combined")
METADATA_CSV = os.path.join(SRC_B, "metadata.csv")

RANDOM_SEED  = 42

TARGETS = {
    "train": {
        "b_real":             7000,
        "b_ai_per_generator": 1400,   # × 5 generators = 7,000
        "c_real":             7000,
        "c_ai":               7000,
    },
    "val": {
        "b_real":             1500,
        "b_ai_per_generator":  300,   # × 5 generators = 1,500
        "c_real":             1500,
        "c_ai":               1500,
    },
}

GENERATORS = [1, 2, 3, 4, 5]

LABEL_B_MAP = {
    1: "SD2.1",
    2: "SDXL",
    3: "SD3",
    4: "DALL-E3",
    5: "MidJourney_v6",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Mounting Google Drive ...")
        drive.mount("/content/drive")
    else:
        print("Google Drive already mounted.")


def _make_dirs() -> None:
    for split in ["train", "val"]:
        for label in ["ai", "real"]:
            os.makedirs(os.path.join(DST_DIR, split, label), exist_ok=True)
    # Empty test/ai and test/real so get_dataloaders does not crash during training
    for label in ["ai", "real"]:
        os.makedirs(os.path.join(DST_DIR, "test", label), exist_ok=True)


def _load_metadata() -> list[dict]:
    records = []
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label_a"] = int(row["label_a"])
            row["label_b"] = int(row["label_b"])
            records.append(row)
    return records


def _list_pngs(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted([p / f for f in os.listdir(folder) if f.lower().endswith(".png")])


def _count(folder: str) -> int:
    return len(_list_pngs(folder))


def _copy_file(src: str, dst: str) -> bool:
    """Copy src to dst. Returns True if copied, False if already existed."""
    if os.path.exists(dst):
        return False
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        print(f"\n  [!] Failed to copy {src}: {exc}")
        return False


def _copy_batch(
    src_paths: list[str],
    dst_folder: str,
    desc: str,
    prefix: str = "",
) -> tuple[int, int]:
    """Copy a list of source files to dst_folder. Returns (copied, skipped).
    prefix is prepended to the filename to avoid collisions between datasets.
    """
    copied = skipped = 0
    for src in tqdm(src_paths, desc=desc, unit="img"):
        dst = os.path.join(dst_folder, prefix + Path(src).name)
        if _copy_file(src, dst):
            copied += 1
        else:
            skipped += 1
    return copied, skipped


# ── SPLIT BUILDER ─────────────────────────────────────────────────────────────

def _build_split(
    split_name: str,
    metadata: list[dict],
    b_split: str,
    c_split: str,
    targets: dict,
) -> None:
    print(f"\n{'─'*60}")
    print(f"  Building: {split_name}")
    print(f"{'─'*60}")

    dst_real = os.path.join(DST_DIR, split_name, "real")
    dst_ai   = os.path.join(DST_DIR, split_name, "ai")

    rng = random.Random(RANDOM_SEED)

    total_copied = total_skipped = 0

    # ── Dataset B — real ──────────────────────────────────────────────────────
    b_real_pool = [
        r for r in metadata
        if r["split"] == b_split and r["label_a"] == 0
    ]
    b_real_selected = rng.sample(b_real_pool, min(targets["b_real"], len(b_real_pool)))
    b_real_paths = [
        os.path.join(SRC_B, r["split"], "real", r["filename"])
        for r in b_real_selected
    ]
    print(f"\n  [B/real]  pool={len(b_real_pool):,}  selected={len(b_real_paths):,}")
    c, s = _copy_batch(b_real_paths, dst_real, desc=f"  {split_name}/real [B]")
    total_copied += c; total_skipped += s

    # ── Dataset B — AI (equal per generator) ──────────────────────────────────
    ai_by_gen: dict[int, list[dict]] = defaultdict(list)
    for r in metadata:
        if r["split"] == b_split and r["label_a"] == 1:
            ai_by_gen[r["label_b"]].append(r)

    b_ai_selected: list[dict] = []
    print(f"\n  [B/ai]    selecting {targets['b_ai_per_generator']} per generator x {len(GENERATORS)} generators:")
    for gen in GENERATORS:
        pool = ai_by_gen.get(gen, [])
        n    = min(targets["b_ai_per_generator"], len(pool))
        if n < targets["b_ai_per_generator"]:
            print(f"  [warn] Generator {gen} ({LABEL_B_MAP[gen]}): only {n} available, needed {targets['b_ai_per_generator']}")
        selected = rng.sample(pool, n)
        b_ai_selected.extend(selected)
        print(f"    gen {gen} ({LABEL_B_MAP[gen]:<16}): pool={len(pool):,}  selected={n}")

    b_ai_paths = [
        os.path.join(SRC_B, r["split"], "ai", r["filename"])
        for r in b_ai_selected
    ]
    print(f"  [B/ai]    total selected: {len(b_ai_paths):,}")
    c, s = _copy_batch(b_ai_paths, dst_ai, desc=f"  {split_name}/ai   [B]")
    total_copied += c; total_skipped += s

    # ── Dataset C — real ──────────────────────────────────────────────────────
    c_real_pool = _list_pngs(os.path.join(SRC_C, c_split, "real"))
    c_real_selected = rng.sample(c_real_pool, min(targets["c_real"], len(c_real_pool)))
    c_real_paths = [str(p) for p in c_real_selected]
    print(f"\n  [C/real]  pool={len(c_real_pool):,}  selected={len(c_real_paths):,}")
    c, s = _copy_batch(c_real_paths, dst_real, desc=f"  {split_name}/real [C]", prefix="c_")
    total_copied += c; total_skipped += s

    # ── Dataset C — AI (random sample) ────────────────────────────────────────
    c_ai_pool = _list_pngs(os.path.join(SRC_C, c_split, "ai"))
    c_ai_selected = rng.sample(c_ai_pool, min(targets["c_ai"], len(c_ai_pool)))
    c_ai_paths = [str(p) for p in c_ai_selected]
    print(f"\n  [C/ai]    pool={len(c_ai_pool):,}  selected={len(c_ai_paths):,}")
    c, s = _copy_batch(c_ai_paths, dst_ai, desc=f"  {split_name}/ai   [C]", prefix="c_")
    total_copied += c; total_skipped += s

    # ── Split summary ─────────────────────────────────────────────────────────
    final_real = _count(dst_real)
    final_ai   = _count(dst_ai)
    print(f"\n  {split_name} complete:")
    print(f"    real:  {final_real:,}  (B: {len(b_real_paths):,}  +  C: {len(c_real_paths):,})")
    print(f"    AI:    {final_ai:,}  (B: {len(b_ai_paths):,}  +  C: {len(c_ai_paths):,})")
    print(f"    total: {final_real + final_ai:,}  |  copied={total_copied:,}  skipped={total_skipped:,}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _mount_drive()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    missing = []
    for path, label in [
        (SRC_B,        "datasets/dataset_b"),
        (SRC_C,        "datasets/dataset_c"),
        (METADATA_CSV, "datasets/dataset_b/metadata.csv"),
    ]:
        if not os.path.exists(path):
            missing.append(f"  x  {label}  ->  {path}")
    if missing:
        raise FileNotFoundError(
            "The following required paths are missing:\n" + "\n".join(missing) + "\n"
            "Run the appropriate download/prepare scripts first."
        )

    print(f"\n{'='*60}")
    print(f"  dataset_combined construction")
    print(f"{'='*60}")
    print(f"  Source B:    {SRC_B}")
    print(f"  Source C:    {SRC_C}")
    print(f"  Destination: {DST_DIR}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"\n  Target sizes:")
    print(f"    train  real: {TARGETS['train']['b_real'] + TARGETS['train']['c_real']:,}"
          f"  (B:{TARGETS['train']['b_real']:,} + C:{TARGETS['train']['c_real']:,})")
    print(f"    train  AI:   {TARGETS['train']['b_ai_per_generator'] * len(GENERATORS) + TARGETS['train']['c_ai']:,}"
          f"  (B:{TARGETS['train']['b_ai_per_generator'] * len(GENERATORS):,} + C:{TARGETS['train']['c_ai']:,})")
    print(f"    val    real: {TARGETS['val']['b_real'] + TARGETS['val']['c_real']:,}"
          f"  (B:{TARGETS['val']['b_real']:,} + C:{TARGETS['val']['c_real']:,})")
    print(f"    val    AI:   {TARGETS['val']['b_ai_per_generator'] * len(GENERATORS) + TARGETS['val']['c_ai']:,}"
          f"  (B:{TARGETS['val']['b_ai_per_generator'] * len(GENERATORS):,} + C:{TARGETS['val']['c_ai']:,})")

    _make_dirs()

    print(f"\n  Loading dataset_b metadata ...")
    metadata = _load_metadata()
    print(f"  Loaded {len(metadata):,} records.")

    # ── Build splits ──────────────────────────────────────────────────────────
    _build_split(
        split_name="train",
        metadata=metadata,
        b_split="train",
        c_split="train",
        targets=TARGETS["train"],
    )

    _build_split(
        split_name="val",
        metadata=metadata,
        b_split="val",
        c_split="val",
        targets=TARGETS["val"],
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE. Final dataset_combined counts:")
    grand_total = 0
    for split in ["train", "val"]:
        for label in ["ai", "real"]:
            folder = os.path.join(DST_DIR, split, label)
            n = _count(folder)
            grand_total += n
            print(f"    {split}/{label}: {n:,}")
    print(f"    grand total: {grand_total:,}")
    print(f"\n  No test split was created.")
    print(f"  Evaluate using --checkpoint_dataset dataset_combined against:")
    print(f"    --dataset dataset_b      (45,000 images, unbalanced diffusion test)")
    print(f"    --dataset dataset_c      (20,000 images, GAN test)")
    print(f"    --dataset ai_vs_human    ( ~8,000 images, full OOD test)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()