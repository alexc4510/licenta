"""
download_dataset_b.py

Downloads the Defactify Image Dataset from HuggingFace, resizes every image
to 224x224 (RGB, center-crop then resize), and saves it directly to Google
Drive so the data persists across Colab sessions.

Run once; re-running is safe — already-saved images are skipped automatically.
"""

import os
import time

from google.colab import drive
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Paste your token here OR set the env-var HF_TOKEN before running.
# Keeping secrets in code is fine for a private Colab notebook, but never
# commit this file to a public repo.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Root on Drive where the processed images will live permanently.
DRIVE_ROOT   = "/content/drive/MyDrive/licenta"
DATASET_DIR  = os.path.join(DRIVE_ROOT, "datasets", "dataset_b")

# Target size after preprocessing
IMG_SIZE = (224, 224)

# How many retries to attempt when a single image save fails
MAX_RETRIES = 3

# Splits present in this dataset
SPLITS = ["train", "validation", "test"]

# Mapping from split name to the subfolder name you want on disk
SPLIT_FOLDER = {
    "train":      "train",
    "validation": "val",
    "test":       "test",
}

# Label_A mapping  (0 = real, 1 = ai-generated)
LABEL_FOLDER = {0: "real", 1: "ai"}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def mount_drive() -> None:
    """Mount Google Drive if it is not already mounted."""
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Mounting Google Drive …")
        drive.mount("/content/drive")
    else:
        print("Google Drive already mounted.")


def make_dirs(base: str) -> None:
    """Create all split/label sub-directories under *base*."""
    for split_folder in SPLIT_FOLDER.values():
        for label_folder in LABEL_FOLDER.values():
            os.makedirs(os.path.join(base, split_folder, label_folder), exist_ok=True)


def preprocess(img: Image.Image) -> Image.Image:
    """
    Convert to RGB and resize to IMG_SIZE using a center-crop strategy:
      1. Scale the shorter side to 224 (preserves aspect ratio).
      2. Center-crop to 224×224.
    This avoids stretching while guaranteeing the exact target resolution.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = IMG_SIZE[0] / min(w, h)
    new_w, new_h = max(IMG_SIZE[0], int(round(w * scale))), max(IMG_SIZE[1], int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Center crop
    left  = (new_w - IMG_SIZE[0]) // 2
    upper = (new_h - IMG_SIZE[1]) // 2
    img   = img.crop((left, upper, left + IMG_SIZE[0], upper + IMG_SIZE[1]))
    return img


def save_with_retry(img: Image.Image, path: str, retries: int = MAX_RETRIES) -> bool:
    """Try to save *img* to *path* up to *retries* times. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            img.save(path, format="PNG", optimize=False)
            return True
        except OSError as exc:
            print(f"    Save attempt {attempt}/{retries} failed: {exc}")
            time.sleep(1.0 * attempt)   # back-off: 1 s, 2 s, 3 s
    return False

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Mount Drive
    mount_drive()

    # 2. Authenticate with HuggingFace
    login(token=HF_TOKEN, add_to_git_credential=False)

    # 3. Prepare directory structure
    make_dirs(DATASET_DIR)

    # 4. Load dataset (streaming=False so we can index freely; HF caches it in
    #    /root/.cache which is on the ephemeral Colab disk — that's fine because
    #    *our* processed images go straight to Drive).
    print("\nLoading dataset from HuggingFace Hub …")
    dataset = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        trust_remote_code=True,
    )

    # 5. Process each split
    total_saved   = 0
    total_skipped = 0
    total_failed  = 0

    for split in SPLITS:
        split_folder = SPLIT_FOLDER[split]
        data         = dataset[split]
        n            = len(data)

        print(f"\n{'─' * 60}")
        print(f"Split : {split}  →  folder '{split_folder}'  ({n} images)")
        print(f"{'─' * 60}")

        saved = skipped = failed = 0

        for i in tqdm(range(n), desc=split, unit="img"):
            # -- Read item --------------------------------------------------
            try:
                item = data[i]
            except Exception as exc:
                print(f"\n  [!] Cannot read item {i}: {exc}")
                failed += 1
                continue

            # -- Validate label --------------------------------------------
            label = item.get("Label_A")
            if label not in LABEL_FOLDER:
                print(f"\n  [!] Unknown Label_A={label!r} at index {i}; skipping.")
                failed += 1
                continue

            # -- Build destination path ------------------------------------
            label_folder = LABEL_FOLDER[label]
            img_path = os.path.join(
                DATASET_DIR, split_folder, label_folder, f"{split}_{i:06d}.png"
            )

            # -- Skip if already done (resumability) -----------------------
            if os.path.exists(img_path):
                skipped += 1
                continue

            # -- Preprocess image ------------------------------------------
            try:
                raw_img = item["Image"]
                if not isinstance(raw_img, Image.Image):
                    raise UnidentifiedImageError("Not a PIL Image object")
                processed = preprocess(raw_img)
            except (UnidentifiedImageError, Exception) as exc:
                print(f"\n  [!] Corrupt/unreadable image at index {i}: {exc}")
                failed += 1
                continue

            # -- Save to Drive ---------------------------------------------
            if save_with_retry(processed, img_path):
                saved += 1
            else:
                print(f"\n  [!] Permanently failed to save index {i}.")
                failed += 1

        print(f"  ✓ saved={saved}  skipped(already existed)={skipped}  failed={failed}")
        total_saved   += saved
        total_skipped += skipped
        total_failed  += failed

    # 6. Summary
    print(f"\n{'═' * 60}")
    print(f"DONE  |  saved={total_saved}  skipped={total_skipped}  failed={total_failed}")
    print(f"Output directory: {DATASET_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()