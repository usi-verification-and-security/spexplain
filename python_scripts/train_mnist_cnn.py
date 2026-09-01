#!/usr/bin/env python3
"""Train a small MNIST CNN (2 conv layers + 1 maxpool) and export it to ONNX.

The exported graph is shaped for spexplain's `explain-onnx`:

* **Flattened input.** `data/datasets/mnist/*.csv` stores each image as 784 flat
  columns while the convolutions need `[1, 1, 28, 28]`. The ONNX input is
  therefore declared image-shaped: `Network2` strips the batch axis and reads
  784 input features, and the CSV columns land on `(1, 28, 28)` in row-major
  order, which is exactly the pixel order the CSV uses. An explicit `Reshape`
  node would be the obvious alternative but `OnnxParser` leaves that case
  unimplemented (`OnnxParser.cpp`), so the FC stage uses `Flatten` instead --
  the same shape the reference `cnn_max_mninst2.onnx` model has.
* **Raw 0-255 pixels.** The CSV holds raw pixel values, but training works far
  better on `[0,1]`. Rather than scale at inference (ONNX `Mul`/`Div` are not
  supported by the encoder), the `1/255` factor is folded into the first
  convolution's weights at export time. That is exact: `W @ (x/255) + b`
  equals `(W/255) @ x + b`, so the exported model consumes raw pixels and the
  input domain is simply [0,255] per feature.
* **No trailing softmax.** The output is the raw logit vector, matching how
  spexplain classifies (argmax over outputs) and staying linear-arithmetic
  encodable.

Only Conv / Relu / MaxPool / Flatten / Gemm nodes are emitted, all of which
`OnnxParser` supports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PIXEL_MAX = 255.0


class MnistCNN(nn.Module):
    """[N,1,28,28] in, 10 logits out."""

    def __init__(self, ch1: int, ch2: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, ch1, kernel_size=4, stride=2)   # 1x28x28 -> ch1x13x13
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, stride=2)  # -> ch2x6x6
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)          # -> ch2x3x3
        self.flatten = nn.Flatten()  # exports as Flatten, not Reshape
        self.fc = nn.Linear(ch2 * 3 * 3, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(self.flatten(x))


def build_loaders(root: str, batch_size: int):
    # ToTensor() already maps the raw bytes into [0,1] and yields [1,28,28].
    tf = transforms.ToTensor()
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
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        acc = accuracy(model, test_loader)
        print(f"  epoch {epoch}/{epochs}  test accuracy {acc:.4f}")
    return acc


def fold_pixel_scale(model: MnistCNN) -> MnistCNN:
    """Return a copy whose conv1 absorbs the 1/255, so it eats raw 0-255 pixels."""
    import copy

    scaled = copy.deepcopy(model)
    with torch.no_grad():
        scaled.conv1.weight.div_(PIXEL_MAX)  # bias is unchanged: W@(x/255)+b == (W/255)@x+b
    return scaled.eval()


def export(model: MnistCNN, onnx_path: Path, opset: int) -> None:
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        # TorchScript exporter: emits the plain Conv/Relu/MaxPool/Gemm graph the
        # OnnxParser understands; the dynamo exporter does not.
        dynamo=False,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="/Users/labbaf/.cache/spexplain-mnist")
    p.add_argument("--out", type=Path, default=Path("data/models/mnist/mnist_cnn.onnx"))
    p.add_argument("--pth", type=Path, default=Path("data/models/mnist/mnist_cnn.pth"))
    p.add_argument("--ch1", type=int, default=4, help="channels out of conv1")
    p.add_argument("--ch2", type=int, default=8, help="channels out of conv2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--opset", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    model = MnistCNN(args.ch1, args.ch2)
    n_relu = args.ch1 * 13 * 13 + args.ch2 * 6 * 6
    print(f"Model: conv({args.ch1}) -> relu -> conv({args.ch2}) -> relu -> maxpool -> fc(10)")
    print(f"  hidden ReLU neurons: {n_relu}   maxpool windows: {args.ch2 * 3 * 3}")

    train_loader, test_loader = build_loaders(args.data_root, args.batch_size)
    acc = train(model, train_loader, test_loader, args.epochs, args.lr)

    args.pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "ch1": args.ch1, "ch2": args.ch2}, args.pth)

    scaled = fold_pixel_scale(model)

    # The folded model on raw pixels must equal the trained model on [0,1] pixels.
    probe = torch.randint(0, 256, (16, 1, 28, 28), dtype=torch.float32)
    with torch.no_grad():
        diff = (scaled(probe) - model(probe / PIXEL_MAX)).abs().max().item()
    print(f"Pixel-scale folding check: max abs diff {diff:.3e}")
    if diff > 1e-3:
        raise SystemExit(f"Folding changed the function (max abs diff {diff:.3e})")

    export(scaled, args.out, args.opset)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Wrote {args.pth} and {args.out} (raw 0-255 pixels, input shape [1,1,28,28] = 784 features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
