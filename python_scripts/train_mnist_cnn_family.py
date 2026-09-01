#!/usr/bin/env python3
"""Train a family of MNIST CNNs of increasing size and export each to ONNX.

Architectures are declared in `ARCHITECTURES` as a list of layer tokens, and the model
name is generated from that list, so the filename describes the structure:

    c8s2-fc              Conv(8ch, stride 2) -> ReLU -> Flatten -> FC(10)
    c4s2-c8s2-p-fc       two convs, then a 2x2 maxpool, then FC(10)
    c16-p-c32-p-f128-fc  two conv+pool stages, a hidden FC(128), then FC(10)

Token grammar (all convs are followed by ReLU; a final FC to 10 classes is implied):

    c<ch>[s<stride>][k<kernel>]   convolution; kernel defaults to 3, stride to 1
    p[<k>]                        maxpool, kernel = stride = k (default 2)
    f<units>                      hidden fully-connected layer + ReLU

Every exported graph follows the same conventions as `train_mnist_cnn.py`, which are what
spexplain's `explain-onnx` needs:

* **Image-shaped input, flat CSV.** The ONNX input is `[1,1,28,28]`; `Network2` strips the
  batch axis to 784 features and the flat CSV columns land on it in row-major order. An
  explicit `Reshape` would be the obvious alternative but `OnnxParser` leaves that case
  unimplemented, so the FC stage uses `Flatten`.
* **Pixel domain, chosen with `--pixel-domain`.** The bounds passed to spexplain must be in
  the same units as the CSV, because spexplain feeds sample values to the network verbatim.
  - `raw255` (default): the `1/255` scaling is folded into the first convolution's weights
    (exact: `W@(x/255)+b == (W/255)@x+b`), so the model eats raw pixels. Use with
    `mnist_quick.csv` / `mnist_short.csv` and bounds `[0,255]`.
  - `unit`: no folding, the model expects already-scaled inputs. Use with
    `mnist_s100_scaled.csv` and bounds `[0,1]`.
  Feeding a model the wrong domain silently produces garbage, not an error.
* **No trailing softmax** -- raw logits, matching how spexplain classifies.

Only Conv / Relu / MaxPool / Flatten / Gemm nodes are emitted.

A `manifest.csv` records each model's shape, parameter count, accuracy, and -- most
importantly for verification cost -- its ReLU-neuron count, which is what drives SMT
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PIXEL_MAX = 255.0
N_CLASSES = 10
INPUT_HW = 28

# Ordered simplest -> largest. Kept deliberately small at the low end: every ReLU becomes
# an SMT variable, and explanation time grows far faster than accuracy does.
ARCHITECTURES: list[list[str]] = [
    ["c4s2k5"],                              # single conv, aggressive stride
    ["c8s2k5"],
    ["c4s2k4", "c8s2"],                      # two convs, no pooling
    ["c4s2k4", "c8s2", "p"],                 # + pooling  (matches train_mnist_cnn.py)
    ["c8", "p", "c16", "p"],                 # classic conv/pool stages
    ["c8s2k4", "c16", "p", "f64"],           # + hidden FC
    ["c16", "p", "c32", "p", "f128"],
    ["c16", "c32", "p", "c32", "p", "f128"],  # three convs, two pools, hidden FC
]

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
    """Built from parsed tokens; input [N,1,28,28], output N x 10 logits."""

    def __init__(self, tokens: list[str]):
        super().__init__()
        layers: list[nn.Module] = []
        ch, hw = 1, INPUT_HW
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
        layers.append(nn.Linear(ch, N_CLASSES))

        self.net = nn.Sequential(*layers)
        self.maxpool_windows = self._count_pool_windows(tokens)

    @staticmethod
    def _count_pool_windows(tokens: list[str]) -> int:
        ch, hw, total = 1, INPUT_HW, 0
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

    def first_conv(self) -> nn.Conv2d:
        return next(m for m in self.net if isinstance(m, nn.Conv2d))


def build_loaders(root: str, batch_size: int):
    tf = transforms.ToTensor()  # yields [1,28,28] in [0,1]
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=1000),
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


def fold_pixel_scale(model: Net) -> Net:
    """Copy whose first conv absorbs 1/255, so the ONNX consumes raw 0-255 pixels."""
    import copy

    scaled = copy.deepcopy(model)
    with torch.no_grad():
        scaled.first_conv().weight.div_(PIXEL_MAX)
    return scaled.eval()


def export(model: Net, onnx_path: Path, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 1, INPUT_HW, INPUT_HW, dtype=torch.float32),
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,  # TorchScript exporter emits the graph OnnxParser understands
    )


def verify_onnx(model: Net, onnx_path: Path, tol: float, pixel_domain: str) -> float:
    """Compare torch vs onnxruntime on in-domain inputs; return max relative diff."""
    import numpy as np

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("verification needs onnxruntime") from exc

    rng = np.random.default_rng(0)
    if pixel_domain == "raw255":
        probe = rng.integers(0, 256, (16, 1, INPUT_HW, INPUT_HW)).astype(np.float32)
    else:
        probe = rng.random((16, 1, INPUT_HW, INPUT_HW)).astype(np.float32)
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


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/models/mnist/cnn"))
    p.add_argument("--data-root", default="/Users/labbaf/.cache/spexplain-mnist")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--opset", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-4, help="relative tolerance for ONNX check")
    p.add_argument(
        "--pixel-domain",
        choices=("raw255", "unit"),
        default="raw255",
        help="raw255: fold 1/255 in, model eats 0-255 pixels. unit: no folding, model eats [0,1]",
    )
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
    p.add_argument("--only", help="regex: train only models whose name matches")
    p.add_argument("--list", action="store_true", help="print the family and exit")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    selected = [t for t in ARCHITECTURES if not args.only or re.search(args.only, model_name(t))]
    if not selected:
        print(f"No architecture matches {args.only!r}", file=sys.stderr)
        return 2

    if args.list:
        print(f"{'name':<34} {'ReLU':>7} {'pool':>6} {'params':>9}")
        for tokens in selected:
            m = Net(tokens)
            n_par = sum(q.numel() for q in m.parameters())
            print(f"{model_name(tokens):<34} {m.relu_neurons:>7} {m.maxpool_windows:>6} {n_par:>9}")
        return 0

    ckpt_dir = args.checkpoint_dir or args.out_dir
    if args.from_checkpoint:
        train_loader = test_loader = None
    else:
        train_loader, test_loader = build_loaders(args.data_root, args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    for i, tokens in enumerate(selected, 1):
        name = model_name(tokens)
        onnx_path = args.out_dir / f"{name}.onnx"
        pth_path = args.out_dir / f"{name}.pth"
        print(f"[{i}/{len(selected)}] {name}", flush=True)

        if onnx_path.exists() and not args.overwrite:
            print("    exists, skipping (use --overwrite)")
            continue

        try:
            torch.manual_seed(args.seed)  # same init seed per model, for comparability
            model = Net(tokens)
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
                acc = train(model, train_loader, test_loader, args.epochs, args.lr)
                secs = time.time() - t0
                torch.save({"model_state_dict": model.state_dict(), "tokens": tokens}, pth_path)

            if args.pixel_domain == "raw255":
                to_export = fold_pixel_scale(model)
                probe = torch.randint(0, 256, (16, 1, INPUT_HW, INPUT_HW), dtype=torch.float32)
                with torch.no_grad():
                    fold_diff = (to_export(probe) - model(probe / PIXEL_MAX)).abs().max().item()
                if fold_diff > 1e-3:
                    raise ValueError(f"pixel-scale folding changed the function ({fold_diff:.3g})")
            else:
                to_export = model.eval()

            export(to_export, onnx_path, args.opset)
            rel = verify_onnx(to_export, onnx_path, args.tol, args.pixel_domain)
            acc_s = "n/a" if acc != acc else f"{acc:.4f}"
            print(f"    acc {acc_s}  train {secs:.0f}s  onnx rel diff {rel:.1e}  -> {onnx_path.name}")

            rows.append({
                "name": name,
                "tokens": " ".join(tokens),
                "pixel_domain": args.pixel_domain,
                "input_min": 0,
                "input_max": 255 if args.pixel_domain == "raw255" else 1,
                "relu_neurons": model.relu_neurons,
                "maxpool_windows": model.maxpool_windows,
                "params": n_par,
                "test_accuracy": "n/a" if acc != acc else f"{acc:.4f}",
                "train_seconds": f"{secs:.0f}",
                "onnx_rel_diff": f"{rel:.2e}",
                "onnx": onnx_path.name,
            })
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"    FAILED: {exc}", file=sys.stderr)

    if rows:
        manifest = args.out_dir / "manifest.csv"
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

    print(f"\n{len(rows)} model(s) trained and exported to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
