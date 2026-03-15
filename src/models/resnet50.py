# src/models/resnet50.py

import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES


def get_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model