#!/usr/bin/env python3
"""Train a family of CNNs of increasing size on MNIST / CIFAR-10 / GTSRB and export to ONNX.

This is the multi-dataset successor to `train_mnist_cnn_family.py`. The architecture grammar
and the ONNX conventions are identical; what is new is that the input shape, class count and
architecture family are selected per dataset:

    dataset    input          classes   csv columns
    mnist      1 x 28 x 28    10        784  + label
    cifar10    3 x 32 x 32    10        3072 + label
    gtsrb      3 x 32 x 32    43        3072 + label

Architectures are declared in `FAMILIES` as lists of layer tokens, and the model name is
generated from the tokens, so the filename describes the structure:

    c8s2k5-fc             Conv(8ch, stride 2, kernel 5) -> ReLU -> Flatten -> FC(classes)
    c16-p-c32-p-fc        two conv+pool stages, then FC(classes)
    c32-p-c64-p-f128-fc   two conv+pool stages, a hidden FC(128), then FC(classes)

Token grammar (all convs are followed by ReLU; a final FC to the class count is implied):

    c<ch>[s<stride>][k<kernel>]   convolution; kernel defaults to 3, stride to 1, no padding
    p[<k>]                        maxpool, kernel = stride = k (default 2)
    f<units>                      hidden fully-connected layer + ReLU

Exported graphs follow the conventions spexplain's `explain-onnx` needs:

* **Image-shaped input, flat CSV.** The ONNX input is `[1,C,H,W]`; `Network2` strips the batch
  axis to C*H*W features and the flat CSV columns land on it in row-major order -- i.e. the CSV
  must be in **CHW** order (all of channel 0 row-major, then channel 1, ...), which is how the
  existing `data/datasets/{cifar,gtsrb}/*_s100_scaled.csv` files are laid out. An explicit
  `Reshape` would be the obvious alternative but `OnnxParser` leaves that case unimplemented,
  so the FC stage uses `Flatten`.
* **Unit pixel domain.** Inputs are plain `ToTensor()` values in `[0,1]`, with no normalisation
  folded in and no mean/std subtraction, so every feature bound passed to spexplain is
  `--input-min 0 --input-max 1`. Feeding a model the wrong domain silently produces garbage,
  not an error.
* **No trailing softmax** -- raw logits, matching how spexplain classifies.

Only Conv / Relu / MaxPool / Flatten / Gemm nodes are emitted.

A `manifest.csv` per output directory records each model's shape, parameter count, accuracy
and -- most importantly for verification cost -- its ReLU-neuron count, which drives SMT
solving time far more than parameter count does.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------

# name -> (channels, side, n_classes, default epochs)
DATASETS = {
    "mnist": (1, 28, 10, 3),
    "cifar10": (3, 32, 10, 12),
    "gtsrb": (3, 32, 43, 10),
}

# Ordered simplest -> largest. Kept deliberately small at the low end: every ReLU becomes an
# SMT variable, and explanation time grows far faster than accuracy does.
MNIST_FAMILY: list[list[str]] = [
    ["c4s2k5"],                               # single conv, aggressive stride
    ["c8s2k5"],
    ["c4s2k4", "c8s2"],                       # two convs, no pooling
    ["c4s2k4", "c8s2", "p"],                  # + pooling
    ["c8", "p", "c16", "p"],                  # classic conv/pool stages
    ["c8s2k4", "c16", "p", "f64"],            # + hidden FC
    ["c16", "p", "c32", "p", "f128"],
    ["c16", "c32", "p", "c32", "p", "f128"],  # three convs, two pools, hidden FC
]

# 32x32x3 inputs carry ~4x the pixels of MNIST, so the same token yields ~4x the ReLUs; the
# family is widened rather than deepened to keep the ReLU counts comparable to the MNIST one.
COLOR_FAMILY: list[list[str]] = [
    ["c8s2k5"],                               # single conv, aggressive stride
    ["c16s2k5"],
    ["c8s2k4", "c16s2"],                      # two convs, no pooling
    ["c8s2k4", "c16s2", "p"],                 # + pooling
    ["c16s2k4", "c32", "p", "f128"],          # + hidden FC
    ["c16", "p", "c32", "p"],                 # classic conv/pool stages
    ["c32", "p", "c64", "p", "f128"],
    ["c32", "c64", "p", "c64", "p", "f256"],  # three convs, two pools, hidden FC
]

FAMILIES = {"mnist": MNIST_FAMILY, "cifar10": COLOR_FAMILY, "gtsrb": COLOR_FAMILY}

_CONV_RE = re.compile(r"^c(?P<ch>\d+)(?:s(?P<s>\d+))?(?:k(?P<k>\d+))?$")
_POOL_RE = re.compile(r"^p(?P<k>\d+)?$")
_FC_RE = re.compile(r"^f(?P<units>\d+)$")


def parse_token(tok: str) -> tuple:
    """Turn one token into ('conv', ch, k, s) / ('pool', k) / ('fc', units)."""
    if m := _CONV_RE.match(tok):
        return ("conv", int(m["ch"]), int(m["k"] or 3), int(m["s"] or 1))
    if m := _POOL_RE.match(tok):
        return ("pool", int(m["k"] or 2))
    if m := _FC_RE.match(tok):
        return ("fc", int(m["units"]))
    raise ValueError(f"Unrecognised architecture token: {tok!r}")


def model_name(tokens: list[str]) -> str:
    return "-".join(tokens) + "-fc"


class Net(nn.Module):
    """Built from parsed tokens; input [N,C,S,S], output N x n_classes logits."""

    def __init__(self, tokens: list[str], in_channels: int, side: int, n_classes: int):
        super().__init__()
        self.in_channels, self.side, self.n_classes = in_channels, side, n_classes
        layers: list[nn.Module] = []
        ch, hw = in_channels, side
        self.relu_neurons = 0

        for spec in (parse_token(t) for t in tokens):
            if spec[0] == "conv":
                _, out_ch, k, s = spec
                hw = (hw - k) // s + 1  # no padding, matching the tested encoder path
                if hw < 1:
                    raise ValueError(f"conv {spec} shrinks the feature map below 1x1")
                layers += [nn.Conv2d(ch, out_ch, kernel_size=k, stride=s), nn.ReLU()]
                ch = out_ch
                self.relu_neurons += out_ch * hw * hw
            elif spec[0] == "pool":
                k = spec[1]
                hw = (hw - k) // k + 1
                if hw < 1:
                    raise ValueError(f"pool {spec} shrinks the feature map below 1x1")
                layers.append(nn.MaxPool2d(kernel_size=k, stride=k))
            else:
                if not any(isinstance(m, nn.Flatten) for m in layers):
                    layers.append(nn.Flatten())
                    ch, hw = ch * hw * hw, 1
                layers += [nn.Linear(ch, spec[1]), nn.ReLU()]
                ch = spec[1]
                self.relu_neurons += spec[1]

        if not any(isinstance(m, nn.Flatten) for m in layers):
            layers.append(nn.Flatten())
            ch = ch * hw * hw
        layers.append(nn.Linear(ch, n_classes))

        self.net = nn.Sequential(*layers)
        self.maxpool_windows = self._count_pool_windows(tokens, in_channels, side)

    @staticmethod
    def _count_pool_windows(tokens: list[str], in_channels: int, side: int) -> int:
        ch, hw, total = in_channels, side, 0
        for spec in (parse_token(t) for t in tokens):
            if spec[0] == "conv":
                _, ch, k, s = spec
                hw = (hw - k) // s + 1
            elif spec[0] == "pool":
                hw = (hw - spec[1]) // spec[1] + 1
                total += ch * hw * hw
        return total

    def forward(self, x):
        return self.net(x)


def raw_datasets(dataset: str, root: str):
    """The torchvision train/test splits, transformed to [C,S,S] tensors in [0,1]."""
    _, side, _, _ = DATASETS[dataset]
    if dataset == "mnist":
        tf = transforms.ToTensor()
        return (datasets.MNIST(root, train=True, download=True, transform=tf),
                datasets.MNIST(root, train=False, download=True, transform=tf))
    if dataset == "cifar10":
        tf = transforms.ToTensor()  # CIFAR-10 is already 32x32
        return (datasets.CIFAR10(root, train=True, download=True, transform=tf),
                datasets.CIFAR10(root, train=False, download=True, transform=tf))
    # gtsrb -- variable-sized crops, resized to a fixed square
    tf = transforms.Compose([transforms.Resize((side, side)), transforms.ToTensor()])
    return (datasets.GTSRB(root, split="train", download=True, transform=tf),
            datasets.GTSRB(root, split="test", download=True, transform=tf))


def materialise(ds, workers: int) -> TensorDataset:
    """Decode a dataset once into a resident tensor pair.

    Every model in the family sees the same fixed (un-augmented) inputs, so decoding and
    resizing per epoch is pure waste -- GTSRB in particular spends far more time in PIL than
    in the network. One pass up front makes the eight-model sweep I/O-free.
    """
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=512, num_workers=workers):
        xs.append(x)
        ys.append(y)
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def build_loaders(dataset: str, root: str, batch_size: int, workers: int = 0):
    """Train/test loaders yielding [C,S,S] tensors in [0,1] and integer labels."""
    train, test = raw_datasets(dataset, root)
    return (
        DataLoader(materialise(train, workers), batch_size=batch_size, shuffle=True),
        DataLoader(materialise(test, workers), batch_size=1000),
    )


@torch.no_grad()
def accuracy(model, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train(model, train_loader, test_loader, epochs: int, lr: float) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        acc = accuracy(model, test_loader)
        print(f"    epoch {epoch}/{epochs}  test acc {acc:.4f}", flush=True)
    return acc


def export(model: Net, onnx_path: Path, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, model.in_channels, model.side, model.side, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,  # TorchScript exporter emits the graph OnnxParser understands
    )


def verify_onnx(model: Net, onnx_path: Path, tol: float) -> float:
    """Compare torch vs onnxruntime on in-domain inputs; return max relative diff."""
    import numpy as np

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("verification needs onnxruntime") from exc

    rng = np.random.default_rng(0)
    probe = rng.random((16, model.in_channels, model.side, model.side)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(probe)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    got = np.concatenate([sess.run(None, {name: p[None]})[0] for p in probe], axis=0)

    scale = np.maximum(np.maximum(np.abs(ref), np.abs(got)), 1.0)
    rel = float(np.max(np.abs(ref - got) / scale))
    if rel > tol:
        raise ValueError(f"torch and onnxruntime disagree: max rel diff {rel:.3g} > {tol:.3g}")
    return rel


def write_sample_csv(dataset: str, loader, path: Path, n: int, seed: int) -> None:
    """Dump `n` test samples as a spexplain CSV: CHW-flattened pixels in [0,1] + label."""
    import numpy as np

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    Y = torch.cat(ys).numpy()
    X = torch.cat(xs).numpy().reshape(len(Y), -1)  # CHW, row-major -- matches Network2
    idx = np.random.default_rng(seed).permutation(len(Y))[:n]  # sample the whole test split

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"px_{i}" for i in range(X.shape[1])] + ["label"])
        for i in idx:
            w.writerow([f"{v:.8g}" for v in X[i]] + [int(Y[i])])
    print(f"Wrote {n} samples to {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--out-dir", type=Path, help="default: data/models/<dataset>/cnn")
    p.add_argument("--data-root", default="/Users/labbaf/.cache/spexplain-data")
    p.add_argument("--epochs", type=int, help="default: per-dataset (see DATASETS)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--opset", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-4, help="relative tolerance for ONNX check")
    p.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="skip training, rebuild from the .pth in --checkpoint-dir and just re-export",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="where to read .pth files for --from-checkpoint (default: --out-dir)",
    )
    p.add_argument("--workers", type=int, default=0,
                   help="dataloader workers for the one-off decode pass (0 = in-process; "
                        "macOS spawn-based workers deadlock here)")
    p.add_argument("--sample-csv", type=Path, help="also dump a spexplain CSV of test samples")
    p.add_argument("--sample-count", type=int, default=100)
    p.add_argument("--only", help="regex: train only models whose name matches")
    p.add_argument("--list", action="store_true", help="print the family and exit")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    in_ch, side, n_classes, default_epochs = DATASETS[args.dataset]
    epochs = args.epochs if args.epochs is not None else default_epochs
    out_dir = args.out_dir or Path(f"data/models/{args.dataset}/cnn")
    family = FAMILIES[args.dataset]

    selected = [t for t in family if not args.only or re.search(args.only, model_name(t))]
    if not selected:
        print(f"No architecture matches {args.only!r}", file=sys.stderr)
        return 2

    if args.list:
        print(f"{'name':<36} {'ReLU':>7} {'pool':>6} {'params':>9}")
        for tokens in selected:
            m = Net(tokens, in_ch, side, n_classes)
            n_par = sum(q.numel() for q in m.parameters())
            print(f"{model_name(tokens):<36} {m.relu_neurons:>7} {m.maxpool_windows:>6} {n_par:>9}")
        return 0

    ckpt_dir = args.checkpoint_dir or out_dir
    if args.from_checkpoint and not args.sample_csv:
        train_loader = test_loader = None
    else:
        train_loader, test_loader = build_loaders(args.dataset, args.data_root, args.batch_size, args.workers)

    if args.sample_csv:
        write_sample_csv(args.dataset, test_loader, args.sample_csv, args.sample_count, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    for i, tokens in enumerate(selected, 1):
        name = model_name(tokens)
        onnx_path = out_dir / f"{name}.onnx"
        pth_path = out_dir / f"{name}.pth"
        print(f"[{i}/{len(selected)}] {args.dataset}/{name}", flush=True)

        if onnx_path.exists() and not args.overwrite:
            print("    exists, skipping (use --overwrite)")
            continue

        try:
            torch.manual_seed(args.seed)  # same init seed per model, for comparability
            model = Net(tokens, in_ch, side, n_classes)
            n_par = sum(q.numel() for q in model.parameters())
            print(f"    ReLU neurons {model.relu_neurons}, maxpool windows "
                  f"{model.maxpool_windows}, params {n_par}", flush=True)

            if args.from_checkpoint:
                src = ckpt_dir / f"{name}.pth"
                if not src.exists():
                    raise FileNotFoundError(f"no checkpoint at {src}")
                model.load_state_dict(torch.load(src, map_location="cpu",
                                                 weights_only=False)["model_state_dict"])
                model.eval()
                acc, secs = float("nan"), 0.0
                print(f"    loaded {src.name}, re-exporting only")
            else:
                t0 = time.time()
                acc = train(model, train_loader, test_loader, epochs, args.lr)
                secs = time.time() - t0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "tokens": tokens,
                    "dataset": args.dataset,
                    "in_channels": in_ch,
                    "side": side,
                    "num_classes": n_classes,
                    "test_accuracy": acc,
                }, pth_path)

            model.eval()
            export(model, onnx_path, args.opset)
            rel = verify_onnx(model, onnx_path, args.tol)
            acc_s = "n/a" if acc != acc else f"{acc:.4f}"
            print(f"    acc {acc_s}  train {secs:.0f}s  onnx rel diff {rel:.1e}"
                  f"  -> {onnx_path.name}")

            rows.append({
                "name": name,
                "tokens": " ".join(tokens),
                "dataset": args.dataset,
                "input_shape": f"{in_ch}x{side}x{side}",
                "num_classes": n_classes,
                "input_min": 0,
                "input_max": 1,
                "relu_neurons": model.relu_neurons,
                "maxpool_windows": model.maxpool_windows,
                "params": n_par,
                "test_accuracy": acc_s,
                "train_seconds": f"{secs:.0f}",
                "onnx_rel_diff": f"{rel:.2e}",
                "onnx": onnx_path.name,
            })
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"    FAILED: {exc}", file=sys.stderr)

    if rows:
        manifest = out_dir / "manifest.csv"
        write_header = not manifest.exists()
        with manifest.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\nManifest: {manifest}")

    if failures:
        print(f"\n{len(failures)} model(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1

    print(f"\n{len(rows)} model(s) trained and exported to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
