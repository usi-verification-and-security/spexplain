#!/usr/bin/env python3
"""ReX script: ResNet-50 (ImageNet-1k), standard occlusion-based mutants.

This is the baseline counterpart to `resnet_ad.py`. It uses exactly the same
model and transforms, but runs an ordinary forward pass on the mutants that
ReX builds by replacing masked pixels with `mask_value`.
"""
import platform

import torch as tt
import torch.nn.functional as F
from PIL import Image  # type: ignore

from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import from_pytorch_tensor

from interpretable_resnet_torchvision import InterpretableResNet50

model = InterpretableResNet50()
model.eval()

if platform.uname().system == "Darwin":
    device = tt.device("mps")
else:
    device = tt.device("cuda")
model.to(device)


def preprocess(path, shape, device, mode) -> Data:
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode="RGB")
    data.data = model.transforms(img).unsqueeze(0).to(device)  # type: ignore
    return data


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        tensor = model(mutants.to(device))
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor, target=target)


def model_shape():
    return ["N", 3, 224, 224]
