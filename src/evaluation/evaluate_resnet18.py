import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from src.models.resnet18 import get_resnet18
from src.data.dataset import get_dataloaders


def evaluate(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def main():

    os.makedirs("metrics", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    _, _, test_loader = get_dataloaders(
        "datasets/ai_vs_human",
        batch_size=64
    )

    model = get_resnet18(num_classes=2).to(device)

    checkpoint_path = "checkpoints/resnet18_epoch_3.pth"

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("Loaded checkpoint:", checkpoint_path)

    preds, labels = evaluate(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print("\nTest Metrics")
    print("---------------------")
    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    metrics = pd.DataFrame({
        "accuracy":[acc],
        "precision":[precision],
        "recall":[recall],
        "f1":[f1]
    })

    metrics_path = "metrics/test_metrics_resnet18.csv"
    metrics.to_csv(metrics_path, index=False)

    print("\nMetrics saved:", metrics_path)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_path = "metrics/confusion_matrix_resnet18.png"
    plt.savefig(cm_path)

    print("Confusion matrix saved:", cm_path)


if __name__ == "__main__":
    main()