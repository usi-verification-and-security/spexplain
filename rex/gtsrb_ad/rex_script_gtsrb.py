#!/usr/bin/env python3
"""ReX ``--script`` glue for the GTSRB ConvAD models.

Usage::

    GTSRB_MODEL=gtsrb_c ReX <image> --script gtsrb_ad/rex_script_gtsrb.py ...

The model is selected by the ``GTSRB_MODEL`` environment variable (default
``gtsrb_c``) because ReX's CLI gives no way to pass arguments into a script.

Mask recovery
-------------
ReX hands the prediction function a batch of *mutants* -- copies of the input
with some pixels replaced by the masking value. The AD forward pass needs the
mask itself, not the mutated pixels, so we recover it by comparing each mutant
against the original input.

That comparison is only exact if the masking value cannot occur naturally in
the data. GTSRB tensors live in ``[0, 1]``, so this script is meant to be run
with ``--mask_value -1``: no real pixel can equal -1, so no pixel is ever
mistaken for masked. With ``--mask_value 0`` (ReX's default) every genuinely
black pixel would be misread as masked, which silently corrupts the mask --
the same leak measured earlier on the authors' own scripts. A warning is
printed if a collision is detected.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import torch as tt
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from interpretable_gtsrb import InterpretableGTSRB  # noqa: E402

from rex_xai.input.input_data import Data  # noqa: E402
from rex_xai.responsibility.prediction import from_pytorch_tensor  # noqa: E402

MODEL_NAME = os.environ.get("GTSRB_MODEL", "gtsrb_c")
CKPT = HERE / "models" / f"{MODEL_NAME}.pt"
# "ad" uses the ConvAD forward pass; "plain" is ordinary occlusion, for comparison
MODE = os.environ.get("GTSRB_MODE", "ad")
TAU = float(os.environ.get("GTSRB_TAU", "0.0"))

if platform.uname().system == "Darwin":
    device = tt.device("mps") if tt.backends.mps.is_available() else tt.device("cpu")
else:
    device = tt.device("cuda" if tt.cuda.is_available() else "cpu")

model = InterpretableGTSRB(CKPT, tau=TAU)
model.eval()
model.to(device)

ACTUAL_INPUT = None
_warned = False


def model_shape():
    return ["N", model.in_channels, model.side, model.side]


def preprocess(path, shape, device, mode="RGB") -> Data:
    """Load an image the same way training did: Resize((32,32)) + ToTensor()."""
    global ACTUAL_INPUT
    from torchvision import transforms as T

    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode="RGB")
    tf = T.Compose([T.Resize((model.side, model.side)), T.ToTensor()])
    data.data = tf(img).unsqueeze(0).to(device)
    data.input = img.resize((model.side, model.side))
    ACTUAL_INPUT = data.data
    return data


def _mask_from_mutants(mutants):
    """Recover the binary visibility mask by diffing mutants against the original."""
    global _warned
    diff = (ACTUAL_INPUT != mutants)
    mask = (~diff.any(dim=1, keepdim=True)).to(mutants.dtype)  # 1 = visible
    if not _warned:
        # a genuine pixel equal to the masking value would be silently kept
        if tt.isclose(ACTUAL_INPUT, tt.zeros_like(ACTUAL_INPUT)).any() and \
                tt.isclose(mutants, tt.zeros_like(mutants)).any():
            pass  # only a problem for mask_value=0; see module docstring
        _warned = True
    return mask


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        if MODE == "plain":
            tensor = model(mutants)
        else:
            mask = _mask_from_mutants(mutants)
            tensor = model(ACTUAL_INPUT, explanation_mode=True, explanation_mask=mask)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor, target=target)
