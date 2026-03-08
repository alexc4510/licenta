import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_PATH = "raw_data/ai_vs_human/versions/4"
IMG_DIR = os.path.join(BASE_PATH, "train_data")

OUTPUT_PATH = "datasets/ai_vs_human"

df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))

df = df[["file_name", "label"]]

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

for split, data in splits.items():

    for label in [0,1]:

        label_name = "real" if label == 0 else "ai"

        folder = os.path.join(OUTPUT_PATH, split, label_name)
        os.makedirs(folder, exist_ok=True)

        subset = data[data["label"] == label]

        for file in tqdm(subset["file_name"], desc=f"{split}-{label_name}"):

            src = os.path.join(BASE_PATH, file)
            dst = os.path.join(folder, os.path.basename(file))

            shutil.copy(src, dst)