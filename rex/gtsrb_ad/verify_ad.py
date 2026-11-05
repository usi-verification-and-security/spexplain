#!/usr/bin/env python3
"""Independent correctness checks for the ConvAD implementation.

Four checks, in increasing strength:

1. **Theorem 4.1** -- with an all-ones mask the AD pass must equal the plain
   network exactly.
2. **Empty mask** -- with an all-zeros mask every conv unit must be deactivated
   (and, since the FC stage then sees a global visible fraction of 0, the FC
   units too).
3. **Monotonicity** -- shrinking the visible region can only ever deactivate
   more units, never fewer.
4. **Receptive field ground truth** -- the real test. For a sampled set of
   hidden units we compute the true receptive field by autograd (which input
   pixels have non-zero gradient w.r.t. that unit) and check that AD marks the
   unit dead exactly when every pixel of that receptive field is masked. This
   validates the mean-value-kernel propagation against something computed a
   completely different way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interpretable_gtsrb import InterpretableGTSRB  # noqa: E402


def check_theorem_41(model, n=8):
    x = torch.rand(n, model.in_channels, model.side, model.side)
    ones = torch.ones(n, 1, model.side, model.side)
    with torch.no_grad():
        d = (model(x) - model(x, explanation_mode=True, explanation_mask=ones)).abs().max()
    return float(d)


def active_counts(model, x, mask, relu_only=True):
    """Active-unit count per layer. The forward pass now returns one mask per
    unit-producing layer, so filter to the ReLU-followed ones by default."""
    with torch.no_grad():
        _, acts = model(x, explanation_mode=True, explanation_mask=mask, return_masks=True)
    if relu_only:
        acts = [acts[i] for i in model.relu_mask_indices()]
    return [int(a.sum()) for a in acts]


def check_empty_mask(model):
    x = torch.rand(1, model.in_channels, model.side, model.side)
    zeros = torch.zeros(1, 1, model.side, model.side)
    return active_counts(model, x, zeros)


def check_monotonic(model, steps=6):
    x = torch.rand(1, model.in_channels, model.side, model.side)
    S = model.side
    prev = None
    ok = True
    seq = []
    for i in range(steps):
        r = S // 2 - i * 2
        m = torch.zeros(1, 1, S, S)
        if r > 0:
            c = S // 2
            m[..., max(0, c - r):c + r, max(0, c - r):c + r] = 1
        tot = sum(active_counts(model, x, m))
        seq.append(tot)
        if prev is not None and tot > prev:
            ok = False
        prev = tot
    return ok, seq


def receptive_field(model, layer_idx, unit_flat_idx, side, in_ch):
    """True receptive field of one hidden unit, via autograd.

    Returns a boolean (side, side) tensor: True where the input pixel can
    influence this unit. Uses a positive input and absolute weights so nothing
    cancels to a spurious zero, and bypasses ReLU gating by taking gradients of
    the pre-activation through a linearised copy.
    """
    lin = _linearised(model)
    x = torch.ones(1, in_ch, side, side, requires_grad=True)
    z = x
    seen = 0
    for layer in lin:
        z = layer(z)
        if isinstance(layer, nn.ReLU):
            if seen == layer_idx:
                break
            seen += 1
    flat = z.reshape(-1)
    flat[unit_flat_idx].backward()
    g = x.grad.abs().sum(dim=1)[0]  # (side, side)
    return g > 0


def _linearised(model):
    """Copy of the stack with |W|, zero bias, and ReLU kept only as a marker.

    Absolute weights guarantee that a zero gradient means 'structurally
    disconnected' rather than 'numerically cancelled'.
    """
    import copy

    layers = []
    for layer in model.net:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            c = copy.deepcopy(layer)
            with torch.no_grad():
                c.weight.abs_()
                if c.bias is not None:
                    c.bias.zero_()
            layers.append(c)
        elif isinstance(layer, nn.MaxPool2d):
            # max-pool is data dependent; average pooling has the same support
            layers.append(nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride))
        else:
            layers.append(layer)
    return nn.Sequential(*layers)


def check_receptive_fields(model, n_units=12, seed=0):
    """AD must kill a unit exactly when its whole receptive field is masked."""
    torch.manual_seed(seed)
    S, C = model.side, model.in_channels
    x = torch.rand(1, C, S, S)
    results = []
    # a few random rectangular masks
    for trial in range(4):
        m = torch.zeros(1, 1, S, S)
        h0, w0 = torch.randint(0, S // 2, (2,)).tolist()
        h1, w1 = (torch.randint(S // 2, S, (2,)) + 1).tolist()
        m[..., h0:h1, w0:w1] = 1
        with torch.no_grad():
            _, acts = model(x, explanation_mode=True, explanation_mask=m, return_masks=True)
        # receptive_field() counts ReLU layers, so index the same subset
        acts = [acts[i] for i in model.relu_mask_indices()]
        # only spatial (conv) layers have a meaningful receptive field test
        for li, a in enumerate(acts):
            if a[0].dim() != 3:
                continue  # FC layer
            flat = a[0].reshape(-1)
            n = flat.numel()
            picks = torch.randint(0, n, (n_units,))
            for u in picks.tolist():
                rf = receptive_field(model, li, u, S, C)
                all_masked = bool((rf & (m[0, 0] > 0)).sum() == 0)
                ad_dead = not bool(flat[u])
                results.append(ad_dead == all_masked)
    return all(results), len(results)


def main():
    for name in sys.argv[1:]:
        ckpt = HERE / "models" / f"{name}.pt" if not name.endswith(".pt") else Path(name)
        model = InterpretableGTSRB(ckpt)
        model.eval()
        print(f"\n=== {ckpt.stem}  {'-'.join(model.tokens)}-fc ===")
        print(f"  hidden layers {model.n_hidden_layers}  sizes {model.hidden_layer_sizes()}"
              f"  total {sum(model.hidden_layer_sizes())} (declared {model.relu_neurons})")

        d = check_theorem_41(model)
        print(f"  [1] Theorem 4.1        max|diff| = {d:.3g}   {'OK' if d == 0 else 'FAIL'}")

        empty = check_empty_mask(model)
        print(f"  [2] empty mask         active per layer = {empty}   "
              f"{'OK' if sum(empty) == 0 else 'FAIL'}")

        ok, seq = check_monotonic(model)
        print(f"  [3] monotonicity       active totals = {seq}   {'OK' if ok else 'FAIL'}")

        ok_rf, n = check_receptive_fields(model)
        print(f"  [4] receptive fields   {n} unit/mask checks   {'OK' if ok_rf else 'FAIL'}")


if __name__ == "__main__":
    main()
