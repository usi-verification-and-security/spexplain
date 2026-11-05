#!/usr/bin/env python3
"""Write spexplain neuron files from ReX + ConvAD explanation masks.

For every explained sample and every hidden ReLU layer (spexplain's numbering,
see ``ad_onnx.py``) three sets are written, each in the
``--input-fix-sample-neuron-activations`` format (sample index line, one line
per hidden layer of 1-based neuron indices terminated by ``0``, blank line):

* ``<model>_ad_deactives.txt`` -- units ConvAD deactivates: the whole receptive
  field lies OUTSIDE the explanation, i.e. the unit reads only free pixels;
* ``<model>_ad_actives.txt``   -- the complement of the deactives;
* ``<model>_ad_inside.txt``    -- units whose whole receptive field lies INSIDE
  the explanation, i.e. the unit reads only pixels an explanation pins to the
  sample. Obtained by the same AD pass with the mask inverted.

Plus ``<model>_ad_meta.json`` with layer sizes/kinds and per-layer means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ad_onnx import ADOnnxModel

HERE = Path(__file__).resolve().parent


def write_sets(path: Path, sample_ids, per_sample_layers):
    with path.open("w") as fh:
        for sid, layers in zip(sample_ids, per_sample_layers):
            fh.write(f"{sid}\n")
            for idx in layers:
                fh.write(" ".join(str(i + 1) for i in idx) + (" 0\n" if idx else "0\n"))
            fh.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .onnx model")
    ap.add_argument("--masks", required=True, type=Path, help="masks .pt from run_rex.py")
    ap.add_argument("--outdir", type=Path, default=HERE / "neuron_activations")
    ap.add_argument("--tau", type=float, default=0.0)
    args = ap.parse_args()

    model = ADOnnxModel(args.model, tau=args.tau)
    name = Path(args.model).stem
    blob = torch.load(args.masks, weights_only=False)
    masks = blob["mask"]  # (S,1,H,W), 1 = explanation pixel
    sample_ids = blob["sample_ids"]

    x = torch.zeros(1, *model.input_shape[1:])  # values irrelevant: only masks are used
    with torch.no_grad():
        _, alive = model.forward_ad(x, masks)
        _, alive_inv = model.forward_ad(x, 1 - masks)

    deact, act, inside = [], [], []
    for s in range(masks.shape[0]):
        deact.append([torch.nonzero(~a[s]).flatten().tolist() for a in alive])
        act.append([torch.nonzero(a[s]).flatten().tolist() for a in alive])
        inside.append([torch.nonzero(~a[s]).flatten().tolist() for a in alive_inv])

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_sets(args.outdir / f"{name}_ad_deactives.txt", sample_ids, deact)
    write_sets(args.outdir / f"{name}_ad_actives.txt", sample_ids, act)
    write_sets(args.outdir / f"{name}_ad_inside.txt", sample_ids, inside)

    sizes = model.hidden_sizes
    mean_deact = [float((~a).float().mean()) for a in alive]
    mean_inside = [float((~a).float().mean()) for a in alive_inv]
    total = sum(sizes)
    meta = {
        "model": name, "gamma": blob["gamma"], "tau": args.tau,
        "n_samples": len(sample_ids), "sample_ids": sample_ids,
        "hidden_sizes": sizes, "hidden_kinds": model.hidden_kinds,
        "mean_deactive_frac_per_layer": mean_deact,
        "mean_inside_frac_per_layer": mean_inside,
        "mean_deactive_frac_total": sum(f * n for f, n in zip(mean_deact, sizes)) / total,
        "mean_inside_frac_total": sum(f * n for f, n in zip(mean_inside, sizes)) / total,
    }
    (args.outdir / f"{name}_ad_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {len(sample_ids)} samples, sizes={sizes}, "
          f"deactive={meta['mean_deactive_frac_total']:.1%} "
          f"inside={meta['mean_inside_frac_total']:.1%} "
          f"per layer={[f'{f:.0%}' for f in mean_deact]}")


if __name__ == "__main__":
    main()
