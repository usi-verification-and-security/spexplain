#!/usr/bin/env python3
"""ReX script: ResNet-50 (ImageNet-1k) with Activation-Deactivation (ConvAD).

Based on the author-supplied `pytorch_script_rex_AD_IN-1k_resnet.py`.
Instead of feeding an occluded image to the model, we recover the binary mask
that ReX used to build the mutant and hand it to the ConvAD forward pass, which
deactivates the corresponding activations inside the network.
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

ACTUAL_INPUT = None


def preprocess(path, shape, device, mode) -> Data:
    global ACTUAL_INPUT
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode="RGB")
    data.data = model.transforms(img).unsqueeze(0).to(device)  # type: ignore
    ACTUAL_INPUT = data.data
    return data


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        mutants = mutants.to(device)
        # a pixel that ReX left untouched is active (1), an occluded one is
        # deactivated (0). This recovers ReX's mask without ever showing the
        # occluded image to the model.
        binary_masks = tt.where(ACTUAL_INPUT != mutants, 0.0, 1.0)
        tensor = model(
            ACTUAL_INPUT, explanation_mask=binary_masks, explanation_mode=True
        )
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor, target=target)


def model_shape():
    return ["N", 3, 224, 224]
