import torch
from src.models.resnet import get_resnet
from src.data.dataset import get_dataloaders

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = get_resnet().to(device)

train_loader, _, _ = get_dataloaders(
    "datasets/ai_vs_human",
    batch_size=8
)

images, labels = next(iter(train_loader))

images = images.to(device)

outputs = model(images)

print("Output shape:", outputs.shape)