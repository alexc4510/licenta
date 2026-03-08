import torch
from src.data.dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    "datasets/ai_vs_human",
    batch_size=8
)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Labels:", labels)