#!/usr/bin/env python3
"""Build `data/models/toy_cnn/toy_cnn.onnx`: a hand-verifiable CNN for testing spexplain's
ONNX / Network2 / OpenSMTVerifier2 path, the way `data/models/toy.nnet` tests the legacy path.

Architecture (integer weights; every intermediate value is computable by hand):

    input [1,1,4,4]  (16 pixels, row-major x0..x15, domain [0,3])
      -> Conv(1,1, kernel=2, stride=2)   4 independent, NON-overlapping 2x2 receptive fields
      -> ReLU
      -> MaxPool(kernel=2, stride=2)     collapses the 2x2 conv output to a single scalar p
      -> Flatten
      -> Gemm (Linear(1,1))              output = -p + 2   (single logit; label = output<0 ? 0 : 1)

Because stride == kernel size, the four receptive fields never overlap: each conv output pixel
depends on exactly 4 distinct pixels, with none shared between the four. This is what keeps the
whole network -- including its SMT-LIB2 encoding -- checkable by hand; see WALKTHROUGH.md.

Conv kernel (applied identically at all 4 positions): [[1, -1], [1, 0]]
  top-left receptive field (x0 x1 / x4 x5): y00 = x0 - x1 + x4 + 0*x5
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = REPO_ROOT / "data/models/toy_cnn/toy_cnn.onnx"
CSV_PATH = REPO_ROOT / "data/datasets/toy_cnn.csv"

# name -> (16 pixels, hand-derived expected label)
SAMPLES = {
    "A_interior_label1": ([0] * 16, 1),
    "B_interior_label0": ([3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    "C_nearboundary_label0": ([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    "D_exactboundary_label1": ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
}


class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1, 1, bias=True)

        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0, -1.0], [1.0, 0.0]]]]))
            self.conv.bias.copy_(torch.tensor([0.0]))
            self.fc.weight.copy_(torch.tensor([[-1.0]]))
            self.fc.bias.copy_(torch.tensor([2.0]))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


def main() -> int:
    model = ToyCNN().eval()

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, 4, 4)
    torch.onnx.export(
        model, dummy, str(ONNX_PATH), input_names=["input"], output_names=["output"],
        opset_version=15, dynamo=False,
    )
    print(f"wrote {ONNX_PATH}")

    rows = []
    for name, (px, expected_label) in SAMPLES.items():
        x = torch.tensor(px, dtype=torch.float32).reshape(1, 1, 4, 4)
        with torch.no_grad():
            out = model(x).item()
        label = 0 if out < 0 else 1
        status = "OK" if label == expected_label else "MISMATCH"
        print(f"  {name:26s} output={out:+.1f}  label={label}  expected={expected_label}  [{status}]")
        assert label == expected_label, f"{name}: hand-derived label does not match model output"
        rows.append(px + [expected_label])

    header = ",".join(f"x{i}" for i in range(16)) + ",output"
    with CSV_PATH.open("w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"wrote {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
