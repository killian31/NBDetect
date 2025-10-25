from typing import Dict

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

LABEL_TO_INDEX: Dict[str, int] = {"not_biting": 0, "biting": 1}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}


def build_model(num_classes: int = 2, pretrained: bool = True, freeze_base: bool = False) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, map_location=None) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    return model
