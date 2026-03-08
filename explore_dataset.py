import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

BASE_PATH = "raw_data/ai_vs_human/versions/4"
TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train_data")

train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

print("TRAIN SHAPE")
print(train.shape)

print("\nTEST SHAPE")
print(test.shape)

print("\nTRAIN HEAD")
print(train.head())

print("\nLABEL DISTRIBUTION")
print(train["label"].value_counts())

train["label"].value_counts().plot(kind="bar")
plt.title("Label distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()


print("\nChecking image sizes...")

sizes = []
sample = train.sample(300)

for f in sample["file_name"]:
    path = os.path.join(BASE_PATH, f)

    try:
        img = Image.open(path)
        sizes.append(img.size)
    except:
        pass

widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

print("\nExample sizes:", sizes[:10])
print("Min width:", min(widths), "Max width:", max(widths))
print("Min height:", min(heights), "Max height:", max(heights))


plt.hist(widths, bins=20)
plt.title("Image width distribution")
plt.xlabel("Width")
plt.ylabel("Frequency")
plt.show()

plt.hist(heights, bins=20)
plt.title("Image height distribution")
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.show()


print("\nShowing random images...")

sample = train.sample(6)

plt.figure(figsize=(10,6))

for i, row in enumerate(sample.itertuples()):

    img_path = os.path.join(BASE_PATH, row.file_name)
    img = Image.open(img_path)

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"Label: {row.label}")
    plt.axis("off")

plt.tight_layout()
plt.show()