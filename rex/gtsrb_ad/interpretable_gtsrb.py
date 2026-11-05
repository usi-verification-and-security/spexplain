#!/usr/bin/env python3
"""ConvAD (Activation-Deactivation) wrapper for the GTSRB CNN family.

Implements the AD forward pass of Chanchal, Kelly and Chockler (arXiv:2510.01038)
for the plain Conv/ReLU/MaxPool/Flatten/Linear stacks produced by
``train_gtsrb_family.py``.

How it works
------------
A binary spatial mask ``M`` (1 = pixel visible, 0 = pixel masked) is carried
alongside the activations. At every *checkpoint* the activations are multiplied
by the current mask, so any unit whose entire receptive field was masked
contributes nothing downstream. The mask is advanced layer by layer:

* **conv** -- convolve the mask with a mean-value kernel of the same geometry.
  The result is the fraction of each output unit's receptive field that is
  still visible; threshold at ``tau`` to get the next mask.
* **maxpool** -- max-pool the mask. A pooled unit stays alive if *any* input in
  its window is alive, which is exactly the ``tau = 0`` rule.
* **flatten + linear** -- a fully connected unit reads the whole feature map, so
  its visible fraction is a single scalar (the global fraction). At ``tau = 0``
  it can only be deactivated when the entire input is masked. This is expected,
  not a bug: AD is about spatial locality and an FC layer has none.

Two deviations from the authors' reference ResNet implementation, both forced
by this architecture:

1. **Bias.** Their code pushes the mask through the conv *module*
   (``bottleneck.conv1(mask)``), which is only sound because torchvision's
   ResNet convs have ``bias=False``. ``nn.Conv2d`` here defaults to
   ``bias=True``, so doing that would add the bias to the mask and it would
   never reach zero. We call ``F.conv2d`` with an explicit ones-kernel and
   ``bias=None`` instead.
2. For the same reason, zeroing the *input* pixels is not by itself faithful --
   a fully masked receptive field would still emit the bias. The explicit
   deactivation step (multiplying by the mask after the conv) is what makes it
   faithful, and it is applied after every conv.

Theorem 4.1 of the paper -- ``N(x) == N'(x, all-ones)`` -- is checked by
``selftest()`` below.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# architecture rebuild (mirrors spexplain's train_cnn_family.Net exactly)
# --------------------------------------------------------------------------
import re

_CONV_RE = re.compile(r"^c(?P<ch>\d+)(?:s(?P<s>\d+))?(?:k(?P<k>\d+))?$")
_POOL_RE = re.compile(r"^p(?P<k>\d+)?$")
_FC_RE = re.compile(r"^f(?P<units>\d+)$")


def parse_token(tok: str) -> tuple:
    if m := _CONV_RE.match(tok):
        return ("conv", int(m["ch"]), int(m["k"] or 3), int(m["s"] or 1))
    if m := _POOL_RE.match(tok):
        return ("pool", int(m["k"] or 2))
    if m := _FC_RE.match(tok):
        return ("fc", int(m["units"]))
    raise ValueError(f"Unrecognised architecture token: {tok!r}")


def build_sequential(tokens, in_channels, side, n_classes):
    layers, ch, hw = [], in_channels, side
    for spec in (parse_token(t) for t in tokens):
        if spec[0] == "conv":
            _, out_ch, k, s = spec
            hw = (hw - k) // s + 1
            layers += [nn.Conv2d(ch, out_ch, kernel_size=k, stride=s), nn.ReLU()]
            ch = out_ch
        elif spec[0] == "pool":
            k = spec[1]
            hw = (hw - k) // k + 1
            layers.append(nn.MaxPool2d(kernel_size=k, stride=k))
        else:
            if not any(isinstance(m, nn.Flatten) for m in layers):
                layers.append(nn.Flatten())
                ch, hw = ch * hw * hw, 1
            layers += [nn.Linear(ch, spec[1]), nn.ReLU()]
            ch = spec[1]
    if not any(isinstance(m, nn.Flatten) for m in layers):
        layers.append(nn.Flatten())
        ch = ch * hw * hw
    layers.append(nn.Linear(ch, n_classes))
    return nn.Sequential(*layers)


class InterpretableGTSRB(nn.Module):
    """Wraps one trained model and adds a ConvAD forward pass.

    ``forward(x)`` is the ordinary network. ``forward(x, explanation_mode=True,
    explanation_mask=M)`` runs the AD pass. With ``return_masks=True`` it also
    returns, for every hidden (ReLU) layer in order, a boolean tensor that is
    True where the unit is still ACTIVE under the mask.
    """

    def __init__(self, ckpt_path: str | Path, tau: float = 0.0):
        super().__init__()
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.tokens = blob["tokens"]
        self.side = blob["side"]
        self.in_channels = blob["in_channels"]
        self.n_classes = blob["n_classes"]
        self.relu_neurons = blob["relu_neurons"]
        self.test_acc = blob.get("acc")
        self.tau = tau
        self.net = build_sequential(
            self.tokens, self.in_channels, self.side, self.n_classes
        )
        # checkpoints come from spexplain's Net, which holds the stack in .net,
        # so the keys arrive prefixed; our self.net IS that Sequential
        sd = blob["state_dict"]
        if any(k.startswith("net.") for k in sd):
            sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
        self.net.load_state_dict(sd)
        self.net.eval()
        self.unit_layers = self._unit_layers()
        self.hidden_shapes = self._hidden_shapes()

    # -- introspection -----------------------------------------------------
    def _unit_layers(self):
        """Describe every unit-producing layer: Conv2d, MaxPool2d, Linear.

        Each entry records whether that layer is immediately followed by a
        ReLU. Flatten is skipped -- it reshapes but produces no new units.

        Only ReLU-followed layers are reported by default (see
        ``neuron_report.py --include-non-relu``); that subset is exactly
        spexplain's notion of a hidden layer.
        """
        info = []
        x = torch.zeros(1, self.in_channels, self.side, self.side)
        layers = list(self.net)
        with torch.no_grad():
            for i, layer in enumerate(layers):
                x = layer(x)
                if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Linear)):
                    nxt = layers[i + 1] if i + 1 < len(layers) else None
                    info.append({
                        "index": i,
                        "kind": type(layer).__name__,
                        "shape": tuple(x.shape[1:]),
                        "followed_by_relu": isinstance(nxt, nn.ReLU),
                    })
        return info

    def _hidden_shapes(self):
        """Shape of every ReLU-followed layer output, in order."""
        return [d["shape"] for d in self.unit_layers if d["followed_by_relu"]]

    @property
    def n_hidden_layers(self) -> int:
        """spexplain's nHiddenLayers: the number of ReLU layers."""
        return len(self.hidden_shapes)

    @staticmethod
    def _numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    def hidden_layer_sizes(self):
        return [self._numel(s) for s in self.hidden_shapes]

    def all_layer_sizes(self):
        """Sizes of every unit-producing layer, ReLU-followed or not."""
        return [self._numel(d["shape"]) for d in self.unit_layers]

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x,
        explanation_mode: bool = False,
        explanation_mask=None,
        masking_value=None,
        tau: float | None = None,
        return_masks: bool = False,
    ):
        if not explanation_mode:
            return self.net(x)

        tau = self.tau if tau is None else tau
        if explanation_mask is None:
            if masking_value is None:
                raise ValueError("explanation_mode needs explanation_mask or masking_value")
            explanation_mask = (x != masking_value).any(dim=1, keepdim=True).float()

        m = explanation_mask
        if m.dim() == 3:
            m = m.unsqueeze(1)
        if m.shape[1] != 1:  # collapse channels: a pixel is visible in all or none
            m = (m > 0).any(dim=1, keepdim=True).float()
        m = m.to(x.dtype)
        if m.shape[0] == 1 and x.shape[0] > 1:
            m = m.expand(x.shape[0], -1, -1, -1)

        # zero the masked input pixels, then deactivate after every conv so the
        # bias of a fully masked receptive field cannot leak through
        z = torch.where(m == 0, torch.zeros_like(x), x)
        masks = []

        def record():
            """Snapshot the current mask, broadcast to this layer's unit shape."""
            if return_masks:
                masks.append((m.expand_as(z) > 0).detach().clone())

        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                z = layer(z)
                k_h, k_w = layer.kernel_size
                ones = torch.ones(1, 1, k_h, k_w, device=m.device, dtype=m.dtype)
                ones = ones / (k_h * k_w)
                # bias=None on purpose -- see module docstring
                frac = F.conv2d(m, ones, bias=None, stride=layer.stride,
                                padding=layer.padding)
                m = (frac > tau).to(z.dtype)
                z = z * m  # checkpoint: kill units whose receptive field is gone
                record()

            elif isinstance(layer, nn.MaxPool2d):
                z = layer(z)
                m = F.max_pool2d(m, kernel_size=layer.kernel_size, stride=layer.stride)
                record()

            elif isinstance(layer, nn.Flatten):
                z = layer(z)
                # spatial structure ends here; carry the global visible fraction
                m = m.flatten(1).mean(dim=1, keepdim=True)

            elif isinstance(layer, nn.Linear):
                z = layer(z)
                gate = (m > tau).to(z.dtype)  # (B,1) -> broadcasts over units
                z = z * gate
                m = gate
                record()

            else:  # ReLU and anything else: shape and mask are unchanged
                z = layer(z)

        if return_masks:
            # one entry per unit-producing layer, aligned with self.unit_layers
            assert len(masks) == len(self.unit_layers), (
                f"recorded {len(masks)} masks for {len(self.unit_layers)} unit layers"
            )
            return z, masks
        return z

    def relu_mask_indices(self):
        """Positions in the returned mask list that are ReLU-followed layers."""
        return [i for i, d in enumerate(self.unit_layers) if d["followed_by_relu"]]


# --------------------------------------------------------------------------
def selftest(ckpt_path, n: int = 8, seed: int = 0) -> dict:
    """Check Theorem 4.1: the AD model with an all-ones mask equals the plain model."""
    torch.manual_seed(seed)
    model = InterpretableGTSRB(ckpt_path)
    x = torch.rand(n, model.in_channels, model.side, model.side)
    with torch.no_grad():
        plain = model(x)
        ones = torch.ones(n, 1, model.side, model.side)
        ad = model(x, explanation_mode=True, explanation_mask=ones)
    diff = (plain - ad).abs().max().item()
    return {
        "max_abs_diff": diff,
        "n_hidden_layers": model.n_hidden_layers,
        "hidden_shapes": model.hidden_shapes,
        "hidden_sizes": model.hidden_layer_sizes(),
        "relu_neurons": model.relu_neurons,
    }


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        r = selftest(p)
        ok = "OK" if r["max_abs_diff"] == 0.0 else "MISMATCH"
        print(f"{Path(p).name:14s} Thm4.1 {ok} (max|diff|={r['max_abs_diff']:.3g})  "
              f"hidden layers={r['n_hidden_layers']} sizes={r['hidden_sizes']} "
              f"sum={sum(r['hidden_sizes'])} (declared {r['relu_neurons']})")
