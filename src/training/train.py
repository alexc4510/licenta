import torch
import torch.nn as nn
import torch.optim as optim

from src.models.resnet import get_resnet
from src.data.dataset import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / (i + 1), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / (i + 1), correct / total


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, _ = get_dataloaders(
        "datasets/ai_vs_human",
        batch_size=16
    )

    model = get_resnet(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2
    max_batches = 50  # limitează pentru sanity check rapid

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, max_batches=max_batches
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, max_batches=20
        )

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "checkpoints/resnet_sanity.pth")
    print("Saved: checkpoints/resnet_sanity.pth")


if __name__ == "__main__":
    main()