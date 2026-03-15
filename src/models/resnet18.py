# src/models/resnet18.py

import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES


def get_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model