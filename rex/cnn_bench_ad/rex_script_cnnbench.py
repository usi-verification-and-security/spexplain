#!/usr/bin/env python3
"""ReX ``--script`` glue for the cnn-bench ONNX models, with ConvAD.

    CNNB_MODEL=<path to .onnx> CNNB_CSV=<dataset csv> ReX s007.png --script rex_script_cnnbench.py ...

ReX needs an image path, but the model input is taken from the dataset CSV
row named by the file (``s007.png`` -> row 7, 1-based), so the explained
values are bit-identical to what spexplain and activation_diff.py see. The PNG
is only used for ReX's rendering.

Run with ``--mask_value -1``: the mask is recovered by diffing mutants against
the original input, and -1 cannot occur in [0,1] data.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as tt
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ad_onnx import ADOnnxModel  # noqa: E402

from rex_xai.input.input_data import Data  # noqa: E402
from rex_xai.responsibility.prediction import from_pytorch_tensor  # noqa: E402

MODEL_PATH = os.environ["CNNB_MODEL"]
CSV_PATH = os.environ["CNNB_CSV"]
MODE = os.environ.get("CNNB_MODE", "ad")  # "plain" = ordinary occlusion
TAU = float(os.environ.get("CNNB_TAU", "0.0"))

device = tt.device("cpu")  # tiny models; MPS dispatch overhead dominates
model = ADOnnxModel(MODEL_PATH, tau=TAU)
model.eval()
_, C, H, W = model.input_shape
FEATURES = pd.read_csv(CSV_PATH).iloc[:, :-1].to_numpy(dtype=np.float32)

ACTUAL_INPUT = None


def model_shape():
    return ["N", C, H, W]


def preprocess(path, shape, device, mode="RGB") -> Data:
    global ACTUAL_INPUT
    row = int(re.search(r"s(\d+)", Path(path).stem).group(1)) - 1
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode="RGB")
    data.data = tt.from_numpy(FEATURES[row]).view(1, C, H, W).to(device)
    data.input = img
    ACTUAL_INPUT = data.data
    return data


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        mutants = mutants.to(device)  # ReX may hand over MPS tensors
        if MODE == "plain":
            tensor = model(mutants)
        else:
            mask = (~(ACTUAL_INPUT != mutants).any(dim=1, keepdim=True)).to(mutants.dtype)
            tensor, _ = model.forward_ad(ACTUAL_INPUT, mask)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor, target=target)
