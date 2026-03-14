from datasets import load_dataset
import os
from tqdm import tqdm

dataset = load_dataset(
    "Rajarshi-Roy-research/Defactify_Image_Dataset"
)

OUTPUT_DIR = "raw_data/dataset_b"

os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = ["train", "validation", "test"]

for split in splits:

    split_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

    data = dataset[split]

    for i in tqdm(range(len(data)), desc=split):

        item = data[i]

        img = item["Image"]
        label = item["Label_A"]

        label_name = "real" if label == 0 else "ai"

        label_dir = os.path.join(split_dir, label_name)

        os.makedirs(label_dir, exist_ok=True)

        img_path = os.path.join(label_dir, f"{split}_{i}.png")

        img.save(img_path)