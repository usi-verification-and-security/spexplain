#!/usr/bin/env python3
"""Run ReX over a set of GTSRB test images and collect the explanation masks.

ReX is driven in-process (patch ``sys.argv``, call ``get_all_args`` then
``explanation``) rather than through the CLI, because we need the sufficiency
mask as an array, not just a rendered PNG.

Two notes on the setup:

* The **patched** ReX checkout at ``<root>/ReX`` is put first on ``sys.path``.
  The copy installed in the conda env still has the five defects found earlier
  (script ``model_shape`` treated as a variable, ``preprocess`` called with the
  wrong arity, missing ``--mask_value`` flag, a visualisation crash for any
  input that is not already model-sized, and ``--analyze`` reading the wrong
  attribute).
* ``--mask_value -1`` is deliberate. The AD glue recovers the mask by diffing a
  mutant against the original, and GTSRB tensors live in ``[0,1]``, so -1
  cannot collide with a real pixel. ReX's default of 0 would make every
  genuinely black pixel look masked.

Writes, per model: a ``.pt`` with the samples and their masks (input to
``neuron_report.py``), a per-image PNG of the explanation, and a CSV of sizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/Users/labbaf/Documents/rex")
HERE = ROOT / "gtsrb_ad"
sys.path.insert(0, str(ROOT / "ReX"))  # patched checkout wins over the installed copy
sys.path.insert(0, str(HERE))

CACHE = Path(
    "/private/tmp/claude-501/-Users-labbaf-Documents-rex/"
    "f6bb35d9-147f-4319-be7c-0c55876ba851/scratchpad"
)


def load_samples(n: int, seed: int):
    """The same n test images the spexplain sample CSV uses, in the same order."""
    blob = torch.load(CACHE / "gtsrb_test_32.pt", weights_only=False)
    X, Y = blob["x"], blob["y"]
    idx_file = HERE / "models" / "gtsrb_test_s100_indices.json"
    if idx_file.exists():
        idx = json.loads(idx_file.read_text())[:n]
    else:
        idx = np.random.default_rng(seed).permutation(len(Y))[:n].tolist()
    return X[idx], Y[idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", default="0.5", help="ReX confidence threshold")
    ap.add_argument("--mask-value", default="-1")
    ap.add_argument("--mode", default="ad", choices=["ad", "plain"])
    ap.add_argument("--iters", default=None)
    ap.add_argument("--outdir", type=Path, default=HERE / "out")
    args = ap.parse_args()

    os.environ["GTSRB_MODEL"] = args.model
    os.environ["GTSRB_MODE"] = args.mode

    from PIL import Image

    from rex_xai.explanation.rex import explanation
    from rex_xai.input.config import get_all_args, validate_args
    from rex_xai.utils._utils import get_device
    from rex_xai.utils.logger import logger, set_log_level

    X, Y, idx = load_samples(args.n, args.seed)
    tag = f"{args.model}_{args.mode}_g{args.gamma}"
    imgdir = args.outdir / "imgs"
    expdir = args.outdir / tag
    imgdir.mkdir(parents=True, exist_ok=True)
    expdir.mkdir(parents=True, exist_ok=True)

    masks, kept, rows = [], [], []
    for i in range(X.shape[0]):
        # ReX takes a path, so write the sample out as a lossless PNG first
        arr = (X[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        png = imgdir / f"s{i + 1:03d}.png"
        Image.fromarray(arr).save(png)

        sys.argv = [
            "ReX", str(png),
            "--script", str(HERE / "rex_script_gtsrb.py"),
            "--mask_value", args.mask_value,
            "--confidence", args.gamma,
            "--output", str(expdir / f"s{i + 1:03d}_exp.png"),
            "-q",
        ]
        if args.iters:
            sys.argv += ["--iters", str(args.iters)]

        t0 = time.time()
        try:
            rex_args = get_all_args()
            validate_args(rex_args)
            set_log_level(rex_args.verbosity, logger)
            exp = explanation(rex_args, get_device(rex_args.gpu), None)
        except Exception as e:  # keep going; record the failure
            print(f"  s{i+1:03d} FAILED {type(e).__name__}: {e}", flush=True)
            rows.append({"sample": i + 1, "dataset_idx": idx[i], "label": int(Y[i]),
                         "status": f"FAILED {type(e).__name__}", "area_px": "",
                         "frac": "", "conf": "", "secs": round(time.time() - t0, 1)})
            continue

        m = exp.sufficiency_mask
        if m is None:
            print(f"  s{i+1:03d} no sufficient explanation found", flush=True)
            rows.append({"sample": i + 1, "dataset_idx": idx[i], "label": int(Y[i]),
                         "status": "no explanation", "area_px": "", "frac": "",
                         "conf": "", "secs": round(time.time() - t0, 1)})
            continue

        m = m.detach().cpu()
        if m.dim() == 3:
            m = m.any(dim=0)          # collapse channels -> (H,W)
        m = m.to(torch.float32).unsqueeze(0)  # (1,H,W)
        masks.append(m)
        kept.append(i)
        area = int(m.sum().item())
        conf = getattr(exp, "sufficiency_confidence", None)
        secs = time.time() - t0
        rows.append({
            "sample": i + 1, "dataset_idx": idx[i], "label": int(Y[i]),
            "status": "ok", "area_px": area,
            "frac": round(area / (m.shape[-1] * m.shape[-2]), 4),
            "conf": round(float(conf), 4) if conf is not None else "",
            "secs": round(secs, 1),
        })
        print(f"  s{i+1:03d} label={int(Y[i]):2d} area={area:4d}px "
              f"({area / 1024:.1%}) conf={conf if conf is None else round(float(conf),3)} "
              f"{secs:.1f}s", flush=True)

    if masks:
        torch.save(
            {"x": X[kept], "y": Y[kept], "mask": torch.stack(masks),
             "dataset_idx": [idx[i] for i in kept],
             "sample_ids": [i + 1 for i in kept],
             "model": args.model, "mode": args.mode, "gamma": args.gamma},
            args.outdir / f"{tag}_masks.pt",
        )
        print(f"saved {len(masks)} masks -> {tag}_masks.pt")

    csv_path = args.outdir / f"{tag}_sizes.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
