import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, resnet50

def get_resnet(num_classes=2):
    model = resnet50(weights=models.ResNet50_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model