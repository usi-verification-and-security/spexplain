#!/usr/bin/env python3
"""Train a family of five GTSRB CNNs and export them for ReX and for spexplain.

Rather than reimplement the architecture/export logic, this imports
``train_cnn_family.py`` from the spexplain repo and reuses its ``Net``,
``export`` and ``verify_onnx``. That guarantees the models obey exactly the
conventions ``explain-onnx`` needs, which are easy to break by accident:

* input ``[1,3,32,32]``, plain ``ToTensor()`` values in ``[0,1]`` -- no
  normalisation, no mean/std, so feature bounds are ``--input-min 0
  --input-max 1``;
* only Conv / Relu / MaxPool / Flatten / Gemm nodes;
* no trailing softmax -- raw logits;
* CSV samples in CHW row-major order, matching ``Network2``.

The five architectures were chosen to hit requested ReLU-neuron counts, which
drive SMT cost far more than parameter count does. Two sit below 9k, one at
roughly 9k, two at roughly 15k, and the set mixes two-conv and three-conv
topologies so the A/D mask can be compared across depths.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
SPEXPLAIN_SCRIPTS = Path(
    "/Users/labbaf/Documents/NN_Explainability/spexplain/neuron-activation/"
    "spexplain/python_scripts"
)
CACHE = Path(
    "/private/tmp/claude-501/-Users-labbaf-Documents-rex/"
    "f6bb35d9-147f-4319-be7c-0c55876ba851/scratchpad"
)

# name -> token list, in the spexplain grammar c<ch>[s<stride>][k<kernel>] / p / f<units>.
# A final Linear to the class count is implied by Net.
ARCHITECTURES: dict[str, list[str]] = {
    "gtsrb_a": ["c8s2k5", "c16", "p", "f64"],                   # 3,936 ReLUs, 2 conv
    "gtsrb_b": ["c12s2k4", "c24", "p", "c32", "p", "f128"],     # 7,396 ReLUs, 3 conv
    "gtsrb_c": ["c16s2k4", "c32", "p", "f128"],                 # 9,136 ReLUs, 2 conv
    "gtsrb_d": ["c12", "p", "c24", "p", "f96"],                 # 14,952 ReLUs, 2 conv
    "gtsrb_e": ["c20s2k5", "c48", "c40", "p", "f128"],          # 14,960 ReLUs, 3 conv
}

N_CLASSES = 43
SIDE = 32
CHANNELS = 3


def load_spexplain_module():
    """Import train_cnn_family.py from the spexplain repo by path."""
    path = SPEXPLAIN_SCRIPTS / "train_cnn_family.py"
    if not path.exists():
        raise FileNotFoundError(f"cannot find {path}")
    spec = importlib.util.spec_from_file_location("spx_train_cnn_family", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cached_loaders(batch_size: int):
    """Loaders over the pre-materialised 32x32 GTSRB tensors.

    The transform is Resize((32,32)) + ToTensor(), identical to what
    ``train_cnn_family.raw_datasets`` builds for gtsrb -- we only cache it so the
    PIL resize is not repeated every epoch.
    """
    out = []
    for split in ("train", "test"):
        blob = torch.load(CACHE / f"gtsrb_{split}_32.pt")
        out.append(TensorDataset(blob["x"], blob["y"]))
    train_ds, test_ds = out
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=512, shuffle=False),
    )


def write_sample_csv(test_loader, path: Path, n: int, seed: int) -> None:
    """Dump n test samples as a spexplain CSV: CHW-flattened pixels in [0,1] + label."""
    import numpy as np

    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
    Y = torch.cat(ys).numpy()
    X = torch.cat(xs).numpy().reshape(len(Y), -1)  # CHW row-major, matches Network2
    idx = np.random.default_rng(seed).permutation(len(Y))[:n]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"px_{i}" for i in range(X.shape[1])] + ["label"])
        for i in idx:
            w.writerow([f"{v:.8g}" for v in X[i]] + [int(Y[i])])
    # the same indices drive the neuron-activation files, so record them
    (path.parent / f"{path.stem}_indices.json").write_text(json.dumps([int(i) for i in idx]))
    print(f"  wrote {n} samples -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=HERE / "models")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    spx = load_spexplain_module()
    train_loader, test_loader = cached_loaders(args.batch_size)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, tokens in ARCHITECTURES.items():
        arch = spx.model_name(tokens)
        print(f"\n=== {name}  {arch} ===", flush=True)
        model = spx.Net(tokens, in_channels=CHANNELS, side=SIDE, n_classes=N_CLASSES)
        n_param = sum(p.numel() for p in model.parameters())
        print(f"  ReLU neurons {model.relu_neurons:,}   params {n_param:,}", flush=True)

        acc = spx.train(model, train_loader, test_loader, args.epochs, args.lr)
        model.eval().cpu()

        pt_path = args.outdir / f"{name}.pt"
        onnx_path = args.outdir / f"{name}.onnx"
        # full module so the ReX/AD wrapper can rebuild it without knowing the tokens
        torch.save(
            {"tokens": tokens, "state_dict": model.state_dict(),
             "in_channels": CHANNELS, "side": SIDE, "n_classes": N_CLASSES,
             "relu_neurons": model.relu_neurons, "acc": acc},
            pt_path,
        )
        spx.export(model, onnx_path, args.opset)
        rel = spx.verify_onnx(model, onnx_path, tol=1e-4)
        print(f"  acc {acc:.4f}   onnx max rel diff {rel:.3g}", flush=True)

        rows.append({
            "name": name, "arch": arch, "tokens": " ".join(tokens),
            "in_channels": CHANNELS, "side": SIDE, "n_classes": N_CLASSES,
            "relu_neurons": model.relu_neurons, "params": n_param,
            "test_acc": round(acc, 4), "onnx_rel_diff": f"{rel:.3g}",
        })

    write_sample_csv(test_loader, args.outdir / "gtsrb_test_s100.csv",
                     args.samples, args.seed)

    man = args.outdir / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {man}")
    for r in rows:
        print(f"  {r['name']:9s} {r['arch']:30s} relu={r['relu_neurons']:6d} acc={r['test_acc']}")


if __name__ == "__main__":
    main()
