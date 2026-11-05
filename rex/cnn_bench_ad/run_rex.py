#!/usr/bin/env python3
"""Run ReX + ConvAD over the rows of a dataset CSV for one cnn-bench model.

Adapted from ``gtsrb_ad/run_rex.py``: ReX is driven in-process (patch
``sys.argv``, call ``get_all_args`` then ``explanation``) because we need the
sufficiency mask as an array. The patched ReX checkout next to this folder is
put first on ``sys.path``.

Writes ``out/<model>_ad_g<gamma>_masks.pt`` (input to ``neuron_files.py``),
``out/<model>_ad_g<gamma>_sizes.csv`` and one explanation PNG per sample.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ReX"))  # patched checkout wins over the installed copy
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .onnx model")
    ap.add_argument("--csv", required=True, help="dataset CSV (features + label)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gamma", default="0.5", help="ReX confidence threshold")
    ap.add_argument("--mask-value", default="-1")
    ap.add_argument("--mode", default="ad", choices=["ad", "plain"])
    ap.add_argument("--outdir", type=Path, default=HERE / "out")
    args = ap.parse_args()

    os.environ["CNNB_MODEL"] = str(Path(args.model).resolve())
    os.environ["CNNB_CSV"] = str(Path(args.csv).resolve())
    os.environ["CNNB_MODE"] = args.mode

    from PIL import Image

    from rex_xai.explanation.rex import explanation
    from rex_xai.input.config import get_all_args, validate_args
    from rex_xai.utils._utils import get_device
    from rex_xai.utils.logger import logger, set_log_level

    df = pd.read_csv(args.csv)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    Y = df.iloc[:, -1].to_numpy()
    n = min(args.n, len(df))
    name = Path(args.model).stem
    side = int(round((X.shape[1] / 3) ** 0.5))

    tag = f"{name}_{args.mode}_g{args.gamma}"
    imgdir = args.outdir / "imgs"
    expdir = args.outdir / tag
    imgdir.mkdir(parents=True, exist_ok=True)
    expdir.mkdir(parents=True, exist_ok=True)

    masks, kept, rows = [], [], []
    for i in range(n):
        png = imgdir / f"s{i + 1:03d}.png"
        if not png.exists():
            arr = (X[i].reshape(3, side, side).transpose(1, 2, 0) * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(png)

        sys.argv = [
            "ReX", str(png),
            "--script", str(HERE / "rex_script_cnnbench.py"),
            "--mask_value", args.mask_value,
            "--confidence", args.gamma,
            "--output", str(expdir / f"s{i + 1:03d}_exp.png"),
            "-q",
        ]

        t0 = time.time()
        row = {"sample": i + 1, "label": int(Y[i]), "status": "", "area_px": "",
               "frac": "", "conf": "", "secs": ""}
        try:
            rex_args = get_all_args()
            rex_args.gpu = False  # the script runs the model on CPU; keep ReX's tensors there too
            validate_args(rex_args)
            set_log_level(rex_args.verbosity, logger)
            exp = explanation(rex_args, get_device(rex_args.gpu), None)
            m = exp.sufficiency_mask
        except Exception as e:  # keep going; record the failure
            row.update(status=f"FAILED {type(e).__name__}", secs=round(time.time() - t0, 1))
            print(f"  s{i+1:03d} FAILED {type(e).__name__}: {e}", flush=True)
            rows.append(row)
            continue

        if m is None:
            row.update(status="no explanation", secs=round(time.time() - t0, 1))
            print(f"  s{i+1:03d} no sufficient explanation found", flush=True)
            rows.append(row)
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
        row.update(status="ok", area_px=area, frac=round(area / m[0].numel(), 4),
                   conf=round(float(conf), 4) if conf is not None else "",
                   secs=round(secs, 1))
        rows.append(row)
        print(f"  s{i+1:03d} label={int(Y[i]):2d} area={area:4d}px "
              f"({area / m[0].numel():.1%}) conf={row['conf']} {secs:.1f}s", flush=True)

    if masks:
        torch.save({"mask": torch.stack(masks), "sample_ids": [i + 1 for i in kept],
                    "model": name, "mode": args.mode, "gamma": args.gamma},
                   args.outdir / f"{tag}_masks.pt")
        print(f"saved {len(masks)} masks -> {tag}_masks.pt")

    with (args.outdir / f"{tag}_sizes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
