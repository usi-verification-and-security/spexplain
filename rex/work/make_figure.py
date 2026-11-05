#!/usr/bin/env python3
"""Montage of ReX explanations: original + each masking strategy + ConvAD."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMAGES = ["ladybird", "peacock", "dog", "starfish", "lizard"]
COLS = [
    ("original", None),
    ("plain zero", "resnet_plain_zero_g0.5"),
    ("plain min", "resnet_plain_min_g0.5"),
    ("plain mean", "resnet_plain_mean_g0.5"),
    ("ConvAD", "resnet_convad_g0.5"),
]

fig, axes = plt.subplots(len(IMAGES), len(COLS), figsize=(3 * len(COLS), 3 * len(IMAGES)))
for r, img in enumerate(IMAGES):
    for c, (title, suffix) in enumerate(COLS):
        ax = axes[r, c]
        path = f"imgs/{img}.jpg" if suffix is None else f"out/{img}_{suffix}.png"
        try:
            ax.imshow(Image.open(path).convert("RGB").resize((224, 224)))
        except FileNotFoundError:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=13)
        if c == 0:
            ax.set_ylabel(img, fontsize=12)

fig.suptitle(
    "ReX minimal sufficient pixel sets, ResNet-50, confidence threshold $\\gamma$=0.5",
    fontsize=15,
)
fig.tight_layout()
fig.savefig("explanation_comparison.png", dpi=110, bbox_inches="tight")
print("wrote explanation_comparison.png")
