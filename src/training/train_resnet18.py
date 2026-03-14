import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

from src.models.resnet18 import get_resnet18
from src.data.dataset import get_dataloaders
from src.config import DATASET_NAME, DATASETS, BATCH_SIZE, EPOCHS, NUM_CLASSES

data_dir = DATASETS[DATASET_NAME]

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for i, (images, labels) in enumerate(loader):

        batch_start = time.time()

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

        if i % 50 == 0:

            elapsed = time.time() - start_time
            batches_done = i + 1
            batches_total = len(loader)

            avg_batch_time = elapsed / batches_done
            remaining = (batches_total - batches_done) * avg_batch_time

            print(
                f"Batch {batches_done}/{batches_total} | "
                f"Loss {loss.item():.4f} | "
                f"Batch time {time.time() - batch_start:.2f}s | "
                f"ETA {remaining/60:.1f} min"
            )

    epoch_loss = total_loss / (i + 1)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):

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

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir,
        batch_size=BATCH_SIZE
    )
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = get_resnet18(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = EPOCHS

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):

        print("\n==============================")
        print(f"Starting Epoch {epoch+1}/{epochs}")
        print("==============================")

        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_time = time.time() - epoch_start

        print("\nEpoch Summary")
        print("------------------------------")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")
        print(f"Epoch time: {epoch_time/60:.2f} minutes")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        torch.save(
            model.state_dict(),
            f"checkpoints/resnet18_epoch_{epoch+1}.pth"
        )

    model_path = "checkpoints/resnet18_baseline.pth"

    torch.save(model.state_dict(), model_path)

    print("\nModel saved:", model_path)

    df = pd.DataFrame(history)

    csv_path = "logs/resnet18_training_metrics.csv"

    df.to_csv(csv_path, index=False)

    print("Metrics saved:", csv_path)

    plt.figure()

    plt.plot(df["epoch"], df["train_loss"], label="Train")
    plt.plot(df["epoch"], df["val_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("ResNet18 Training vs Validation Loss")

    plt.legend()

    plt.savefig("logs/resnet18_loss_curve.png")

    plt.figure()

    plt.plot(df["epoch"], df["train_acc"], label="Train")
    plt.plot(df["epoch"], df["val_acc"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title("ResNet18 Training vs Validation Accuracy")

    plt.legend()

    plt.savefig("logs/resnet18_accuracy_curve.png")

    print("Plots saved in logs/")


if __name__ == "__main__":
    main()