import os
import shutil

SRC = "raw_data/dataset_b"
DST = "datasets/dataset_b"

mapping = {
    "train": "train",
    "validation": "val",
    "test": "test"
}

for src_split, dst_split in mapping.items():

    for label in ["real", "ai"]:

        src_dir = os.path.join(SRC, src_split, label)
        dst_dir = os.path.join(DST, dst_split, label)

        os.makedirs(dst_dir, exist_ok=True)

        for f in os.listdir(src_dir):

            src_file = os.path.join(src_dir, f)
            dst_file = os.path.join(dst_dir, f)

            shutil.copy(src_file, dst_file)