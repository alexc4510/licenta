# scripts/download_dataset_a.py
"""
Downloads the ai-vs-human-generated-dataset from Kaggle and copies it to
Google Drive for persistence across Colab sessions.

Prerequisites
─────────────
Your Kaggle credentials must be available.  Either:
  • Place  ~/.kaggle/kaggle.json  in the Colab environment, OR
  • Set env vars:  KAGGLE_USERNAME  and  KAGGLE_KEY

Run once.  Re-running is safe — the Drive copy is skipped if already present.
"""

import os
import shutil

from google.colab import drive

DRIVE_ROOT = "/content/drive/MyDrive/licenta"
RAW_DST    = os.path.join(DRIVE_ROOT, "raw_data", "ai_vs_human")


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Mounting Google Drive …")
        drive.mount("/content/drive")
    else:
        print("Google Drive already mounted.")


def main() -> None:
    _mount_drive()

    print("\nDownloading dataset from Kaggle …")
    import kagglehub
    cache_path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")
    print(f"Kaggle cache path: {cache_path}")

    if os.path.isdir(RAW_DST):
        print(f"\nDrive destination already exists — skipping copy:\n  {RAW_DST}")
    else:
        print(f"\nCopying to Drive …\n  {cache_path}\n  → {RAW_DST}")
        shutil.copytree(cache_path, RAW_DST)
        print("Copy complete.")

    print(f"\nRaw data on Drive: {RAW_DST}")
    print("Next step: python scripts/prepare_dataset_a.py")


if __name__ == "__main__":
    main()