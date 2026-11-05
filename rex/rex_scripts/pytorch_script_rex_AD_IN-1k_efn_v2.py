#!/usr/bin/env python3
import sys

import platform
import torch as tt
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image  # type: ignore
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import from_pytorch_tensor
import matplotlib.pyplot as plt
from interpretable_efficientnetv2 import InterpretableEfficientNetV2
import numpy as np

model = InterpretableEfficientNetV2()
model.eval()

device = None
ACTUAL_INPUT = None

if platform.uname().system == "Darwin":
    device = tt.device("mps")
    model.to("mps")
else:
    device = tt.device("cuda:0")
    model.to("cuda:0")


def preprocess(path, shape, device, mode) -> Data:
    global ACTUAL_INPUT
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode='RGB')
    data.data = model.transforms(img).unsqueeze(0).to(device)  # type: ignore
    ACTUAL_INPUT = data.data

    return data


def prediction_function(mutants, masks_objects=None, 
                        target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        binary_masks = tt.where(ACTUAL_INPUT != mutants, 0.0, 1.0)
        tensor = model(ACTUAL_INPUT, explanation_mask=binary_masks,
                    explanation_mode=True)
        if raw:
            return F.softmax(tensor, dim=1)
        
        return from_pytorch_tensor(tensor, target=target)


def model_shape():
    return ["N", 3, 300, 300]
