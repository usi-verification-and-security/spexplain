#!/usr/bin/env python3
"""Baseline neuron files for spexplain's hybrid mode, to compare against the heuristics.

For every model and every sample of the dataset, writes three
``--input-fix-sample-neuron-activations`` files (same format and numbering as
``activation_diff.py``: sample index line, one line per hidden ReLU layer of
1-based neuron indices terminated by ``0``, blank line):

* ``random/<model>.txt``   -- a uniformly random subset of each layer, of exactly
  the size the adversarial-search file fixes for that sample and layer (seeded).
  Samples the adversarial file skips (PGD found no counterfactual) are skipped
  here too, so the two differ only in *which* neurons are fixed;
* ``negative/<model>.txt`` -- neurons whose pre-activation at the sample is < 0
  (inactive ReLUs);
* ``positive/<model>.txt`` -- neurons whose pre-activation at the sample is > 0
  (active ReLUs).

Pre-activations exactly 0 belong to neither set. Neurons are numbered by the
same ONNX interpreter ``activation_diff.py`` uses, i.e. exactly as spexplain.

Usage:
    python make_baseline_files.py <model.onnx>... --csv <dataset.csv> \\
        --adv-dir <dir with adversarial <model>.txt> --outdir <data/activations/<type>>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_torch_model import OnnxTorchModel  # noqa: E402


class PreActModel(OnnxTorchModel):
    """Also records the input of every Relu node (the pre-activation)."""

    def forward(self, x):
        self.pre_activations = []
        return super().forward(x)

    def _run(self, op, ins, attrs):
        if op == "Relu":
            self.pre_activations.append(ins[0])
        return super()._run(op, ins, attrs)


def parse(path: Path, n_layers: int) -> dict[int, list[list[int]]]:
    """spexplain's reader: sample index, then per hidden layer 1-based indices up to 0."""
    toks = [int(t) for t in path.read_text().split()]
    pos, out = 0, {}
    while pos < len(toks):
        sid = toks[pos]; pos += 1
        layers = []
        for _ in range(n_layers):
            cur = []
            while toks[pos] != 0:
                cur.append(toks[pos]); pos += 1
            pos += 1
            layers.append(cur)
        out[sid] = layers
    return out


def write(path: Path, records: dict[int, list[list[int]]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for sid in sorted(records):
            fh.write(f"{sid}\n")
            for idx in records[sid]:
                fh.write(" ".join(map(str, idx)) + (" 0\n" if idx else "0\n"))
            fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--adv-dir", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = pd.read_csv(args.csv).iloc[:, :-1].to_numpy(dtype=np.float32)

    for path in args.models:
        name = path.stem
        model = PreActModel(str(path))
        model.eval()
        with torch.no_grad():
            model(torch.from_numpy(X).view(len(X), *model.input_shape[1:]))
        pre = [p.reshape(len(X), -1) for p in model.pre_activations]
        sizes = [p.shape[1] for p in pre]

        neg = {s + 1: [(torch.nonzero(p[s] < 0).flatten() + 1).tolist() for p in pre]
               for s in range(len(X))}
        pos = {s + 1: [(torch.nonzero(p[s] > 0).flatten() + 1).tolist() for p in pre]
               for s in range(len(X))}

        adv = parse(args.adv_dir / f"{name}.txt", len(sizes))
        rng = np.random.default_rng([args.seed, sum(map(ord, name))])  # per-model stream
        rnd = {}
        for sid, layers in adv.items():
            rnd[sid] = [sorted((rng.choice(n, size=len(fixed), replace=False) + 1).tolist())
                        for n, fixed in zip(sizes, layers)]

        write(args.outdir / "negative" / f"{name}.txt", neg)
        write(args.outdir / "positive" / f"{name}.txt", pos)
        write(args.outdir / "random" / f"{name}.txt", rnd)

        def frac(rec):
            return sum(len(l) for r in rec.values() for l in r) / (len(rec) * sum(sizes))
        print(f"{name}: sizes={sizes} samples={len(X)} "
              f"negative={frac(neg):.1%} positive={frac(pos):.1%} "
              f"random={frac(rnd):.1%} ({len(rnd)} samples, adversarial={frac(adv):.1%})")


if __name__ == "__main__":
    main()
