#!/usr/bin/env python3
"""Sanity checks on the ConvAD ResNet-50 implementation.

1. Theorem 4.1: with an all-ones mask, N'(x, 1) == N(x).
2. Deactivation is not the same as zero-occlusion: for a partial mask the AD
   forward pass should differ from feeding the occluded image to the plain model.
3. AD is insensitive to the occlusion value used to build the mutant, while the
   plain forward pass is not.
"""
import sys

import torch as tt
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/Users/labbaf/Documents/rex/models")
from interpretable_resnet_torchvision import InterpretableResNet50

device = tt.device("mps")
model = InterpretableResNet50().eval().to(device)

img = Image.open("/Users/labbaf/Documents/rex/work/imgs/ladybird.jpg").convert("RGB")
x = model.transforms(img).unsqueeze(0).to(device)

with tt.no_grad():
    plain = F.softmax(model(x), dim=1)
    ones = tt.ones_like(x)
    ad_full = F.softmax(model(x, explanation_mask=ones, explanation_mode=True), dim=1)

top = plain.argmax().item()
print(f"[1] Theorem 4.1 (empty mask)")
print(f"    plain  top={top} conf={plain[0, top]:.6f}")
print(f"    AD(1)  top={ad_full.argmax().item()} conf={ad_full[0, top]:.6f}")
print(f"    max |diff| = {(plain - ad_full).abs().max().item():.3e}")

# a centre patch mask: keep the middle 80x80, deactivate the rest
mask = tt.zeros_like(x)
mask[:, :, 72:152, 72:152] = 1.0

with tt.no_grad():
    ad_part = F.softmax(model(x, explanation_mask=mask, explanation_mode=True), dim=1)
    print(f"\n[2] partial mask (centre 80x80 kept), target class {top}")
    print(f"    AD           conf={ad_part[0, top]:.6f} top={ad_part.argmax().item()}")

    for name, val in [("zero", 0.0), ("min", x.min().item()), ("max", x.max().item()),
                      ("mean", x.mean().item())]:
        occluded = tt.where(mask.bool(), x, tt.full_like(x, val))
        p = F.softmax(model(occluded), dim=1)
        print(f"    occl {name:5s} conf={p[0, top]:.6f} top={p.argmax().item()}")

print("\n[3] AD is invariant to the occlusion value used to build the mutant")
with tt.no_grad():
    for name, val in [("zero", 0.0), ("min", x.min().item()), ("max", x.max().item())]:
        mutant = tt.where(mask.bool(), x, tt.full_like(x, val))
        recovered = tt.where(x != mutant, 0.0, 1.0)
        p = F.softmax(model(x, explanation_mask=recovered, explanation_mode=True), dim=1)
        print(f"    mutant built with {name:5s} -> AD conf={p[0, top]:.6f}")
