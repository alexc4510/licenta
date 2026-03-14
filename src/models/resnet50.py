import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES

def get_resnet50(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model