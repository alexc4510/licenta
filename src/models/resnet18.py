import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def get_resnet18(num_classes=NUM_CLASSES):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model