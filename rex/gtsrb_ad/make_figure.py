#!/usr/bin/env python3
"""Montage of GTSRB inputs and their ReX+ConvAD explanations.

32x32 images are unreadable at native size, so everything is upscaled with
nearest-neighbour (no interpolation, so pixel boundaries stay honest).
Row 1: input. Row 2: the AD mask. Row 3: input with non-explanation pixels dimmed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gtsrb_a")
    ap.add_argument("--gamma", default="0.5")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    blob = torch.load(HERE / "out" / f"{args.model}_ad_g{args.gamma}_masks.pt",
                      weights_only=False)
    X, M, Y = blob["x"], blob["mask"], blob["y"]
    ids = blob["sample_ids"]
    n = min(args.n, X.shape[0])

    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5.2))
    for j in range(n):
        img = X[j].permute(1, 2, 0).numpy()
        m = M[j, 0].numpy()
        axes[0, j].imshow(img, interpolation="nearest")
        axes[0, j].set_title(f"s{ids[j]:03d} cls {int(Y[j])}", fontsize=8)
        axes[1, j].imshow(m, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].set_title(f"{int(m.sum())}px ({m.mean():.1%})", fontsize=8)
        dim = img * (0.15 + 0.85 * m[..., None])
        axes[2, j].imshow(np.clip(dim, 0, 1), interpolation="nearest")
        for r in range(3):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    for r, lab in enumerate(["input", "AD mask", "explanation"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)

    fig.suptitle(
        f"ReX + ConvAD explanations, {args.model} (gamma={args.gamma})", fontsize=12)
    fig.tight_layout()
    out = args.out or HERE / "out" / f"{args.model}_explanations.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
