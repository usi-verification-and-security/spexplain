#!/usr/bin/env python3
"""Extract MSPS masks with ReX, then score them all under ONE common protocol.

Why this exists: ReX's built-in `--analyse` robustness calls the script's own
prediction function. For the AD script that function deactivates everything
outside the mask, so the background it is supposedly being tested against is
invisible to the model and robustness is 1.0 by construction. To compare the
methods fairly we re-evaluate every mask the same way, on the *unmodified*
ResNet-50, by pasting the MSPS onto real backgrounds (the paper's Def 5.1).
"""
import itertools
import os
import sys

import numpy as np
import torch as tt
import torch.nn.functional as F
from PIL import Image

ROOT = "/Users/labbaf/Documents/rex"
sys.path.insert(0, f"{ROOT}/ReX")
sys.path.insert(0, f"{ROOT}/models")

from rex_xai.input.config import get_all_args, validate_args
from rex_xai.explanation.rex import explanation
from rex_xai.utils._utils import get_device
from rex_xai.utils.logger import logger, set_log_level

IMAGES = ["ladybird", "peacock", "dog", "starfish", "lizard"]
METHODS = {
    "plain_zero": ("scripts/resnet_plain.py", "0"),
    "plain_min": ("scripts/resnet_plain.py", "min"),
    "plain_mean": ("scripts/resnet_plain.py", "mean"),
    "convad": ("scripts/resnet_ad.py", "0"),
}
GAMMA = "0.5"
MASKDIR = f"{ROOT}/work/masks"


def get_mask(img, method):
    """Run ReX and return its sufficiency mask, caching to disk."""
    os.makedirs(MASKDIR, exist_ok=True)
    out = f"{MASKDIR}/{img}_{method}_g{GAMMA}.npy"
    if os.path.exists(out):
        return np.load(out)

    script, mv = METHODS[method]
    sys.argv = [
        "ReX", f"imgs/{img}.jpg",
        "--script", script,
        "--mask_value", mv,
        "--confidence", GAMMA,
        "-q",
    ]
    args = get_all_args()
    validate_args(args)
    set_log_level(args.verbosity, logger)
    device = get_device(args.gpu)
    exp = explanation(args, device, None)
    if exp.sufficiency_mask is None:
        return None
    mask = exp.sufficiency_mask.detach().cpu().numpy()
    np.save(out, mask)
    return mask


def main():
    os.chdir(f"{ROOT}/work")

    # collect masks first (each needs its own model in memory, so do it up front)
    masks = {}
    for img, method in itertools.product(IMAGES, METHODS):
        print(f"extracting {img}/{method} ...", flush=True)
        try:
            m = get_mask(img, method)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}", flush=True)
            m = None
        if m is not None:
            masks[(img, method)] = m
            print(f"  {int(np.count_nonzero(m) / 3)} px", flush=True)

    # now score them all on a clean, unmodified ResNet-50
    from interpretable_resnet_torchvision import InterpretableResNet50

    device = tt.device("mps")
    model = InterpretableResNet50().eval().to(device)

    # backgrounds: solid colours spanning the data range + the other images
    def backgrounds(x):
        lo, hi = x.min().item(), x.max().item()
        bgs = [tt.full_like(x, v) for v in np.linspace(lo, hi, 8)]
        for other in IMAGES:
            o = model.transforms(
                Image.open(f"imgs/{other}.jpg").convert("RGB")
            ).unsqueeze(0).to(device)
            bgs.append(o)
        # plus random noise
        for seed in range(4):
            g = tt.Generator(device="cpu").manual_seed(seed)
            bgs.append(tt.empty(x.shape).uniform_(lo, hi, generator=g).to(device))
        return bgs

    print("\n=== fair robustness: MSPS pasted onto backgrounds, clean model ===")
    print(f"{'image':10s} {'method':12s} {'px':>6s} {'own conf':>9s} {'retained':>9s}")
    rows = []
    for img in IMAGES:
        x = model.transforms(
            Image.open(f"imgs/{img}.jpg").convert("RGB")
        ).unsqueeze(0).to(device)
        with tt.no_grad():
            target = F.softmax(model(x), dim=1).argmax().item()
        bgs = backgrounds(x)
        for method in METHODS:
            m = masks.get((img, method))
            if m is None:
                continue
            mask = tt.from_numpy(m).to(device).unsqueeze(0)
            with tt.no_grad():
                own = F.softmax(
                    model(tt.where(mask, x, tt.zeros_like(x))), dim=1
                )[0, target].item()
                good = 0
                for bg in bgs:
                    composite = tt.where(mask, x, bg)
                    p = F.softmax(model(composite), dim=1)
                    if p.argmax().item() == target:
                        good += 1
            px = int(np.count_nonzero(m) / 3)
            rate = good / len(bgs)
            rows.append((img, method, px, own, rate))
            print(f"{img:10s} {method:12s} {px:6d} {own:9.4f} {rate:9.2f}")

    print("\n=== mean over images ===")
    for method in METHODS:
        sel = [r for r in rows if r[1] == method]
        if sel:
            print(
                f"{method:12s} px={np.mean([r[2] for r in sel]):7.1f} "
                f"retained={np.mean([r[4] for r in sel]):.3f}"
            )

    import csv
    with open("fair_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "method", "px", "own_conf_black_bg", "retained_frac"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
