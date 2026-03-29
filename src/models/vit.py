# src/models/vit.py

import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES


def get_vit(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    ViT-B/16 with pretrained ImageNet weights.
    The classification head is replaced with a new linear layer
    matching the number of output classes.

    Architecture: Vision Transformer Base with 16x16 patch size.
    Parameters: ~86 million
    Input resolution: 224x224 (same as ResNet models, no preprocessing changes needed)
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model