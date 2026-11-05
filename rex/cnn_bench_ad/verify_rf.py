#!/usr/bin/env python3
"""Independent check of the ConvAD receptive-field logic in ad_onnx.py.

With every weight replaced by |W|, biases dropped and ReLUs turned into the
identity, feeding the visibility mask itself through the network gives a unit
value > 0 exactly when some visible pixel lies in that unit's receptive field.
So ConvAD's dead set (tau = 0) must equal the set of units whose value is 0.
"""
import sys
import torch
import torch.nn.functional as F
from ad_onnx import ADOnnxModel


def rf_dead(model, mask):
    weights = {}
    for n in model.nodes:
        if n.op_type in ("Conv", "Gemm", "MatMul"):
            weights[n.input[1]] = "abs"
            if len(n.input) > 2:
                weights[n.input[2]] = "zero"
    saved = {}
    for name, how in weights.items():
        buf = model._init(name)
        saved[name] = buf.clone()
        buf.copy_(buf.abs() if how == "abs" else torch.zeros_like(buf))
    relu = F.relu
    try:
        F.relu = lambda t: t  # onnx_torch_model calls F.relu
        x = mask.expand(-1, model.input_shape[1], -1, -1).double()
        model.double()
        with torch.no_grad():
            model(x)
        return [(a.reshape(a.shape[0], -1) == 0) for a in model.relu_activations]
    finally:
        F.relu = relu
        for name, buf in saved.items():
            model._init(name).copy_(buf)
        model.float()


for path in sys.argv[1:]:
    torch.manual_seed(0)
    model = ADOnnxModel(path)
    _, _, H, W = model.input_shape
    # random rectangles plus random speckle, 32 masks
    masks = (torch.rand(32, 1, H, W) < 0.15).float()
    for i in range(16):
        r, c = torch.randint(0, H - 8, (2,))
        masks[i] = 0
        masks[i, 0, r:r + 8, c:c + 10] = 1
    with torch.no_grad():
        _, alive = model.forward_ad(torch.zeros(1, *model.input_shape[1:]), masks)
    ref = rf_dead(model, masks)
    ok = all(torch.equal(~a, r) for a, r in zip(alive, ref))
    print(f"{path.split('/')[-1]:10s} {'OK' if ok else 'MISMATCH'} "
          f"dead fractions {[round(float(r.float().mean()), 3) for r in ref]}")
