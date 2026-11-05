#!/usr/bin/env python3
"""ConvAD (Activation-Deactivation) on an ONNX graph, for the cnn-bench models.

Extends ``OnnxTorchModel`` (copied verbatim from spexplain's
``python_scripts/contrastive-guide/onnx_torch_model.py``, the loader that
``activation_diff.py`` uses to write the adversarial-search neuron files).
Running ConvAD on the very same interpreter guarantees that the k-th ReLU node
here is spexplain's hidden layer k, and that neurons are numbered in the same
flatten order -- so the AD neuron files and the adversarial ones are directly
comparable, index by index.

AD forward pass (Chanchal, Kelly, Chockler, arXiv:2510.01038, tau = 0 default)
-----------------------------------------------------------------------------
A binary spatial mask (1 = pixel visible) travels alongside every tensor:

* input -- masked pixels are zeroed;
* Conv -- convolve the mask with a mean-value kernel of the same geometry (the
  visible fraction of each unit's receptive field), threshold at tau, and
  multiply the activations by the result. The multiplication is what keeps
  the bias of a fully masked receptive field from leaking through;
* MaxPool / AveragePool -- same rule with the pooling window: a pooled unit
  is alive if more than tau of its window is alive (at tau = 0: any of it);
* Relu -- mask unchanged; this is where the per-hidden-layer mask is recorded;
* Flatten / Reshape -- spatial structure ends; the mask collapses to the
  global visible fraction of the feature map;
* Gemm / MatMul -- a fully connected unit reads the whole feature map, so it
  is deactivated only when that global fraction is <= tau (at tau = 0: only
  when everything is masked).

``selftest`` checks Theorem 4.1 (all-ones mask == plain network) and that the
recorded masks align with the plain pass's ReLU activations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_torch_model import OnnxTorchModel  # noqa: E402


class ADOnnxModel(OnnxTorchModel):

    def __init__(self, onnx_path, tau: float = 0.0):
        super().__init__(onnx_path)
        self.tau = tau
        x = torch.zeros(1, *self.input_shape[1:])
        with torch.no_grad():
            self(x)
        self.hidden_shapes = [tuple(a.shape[1:]) for a in self.relu_activations]
        self.hidden_sizes = [int(a[0].numel()) for a in self.relu_activations]
        self.hidden_kinds = self._hidden_kinds()

    def _hidden_kinds(self):
        """'conv' or 'fc' for each ReLU layer, from the op that feeds it."""
        producer = {n.output[0]: n for n in self.nodes}
        kinds = []
        for n in self.nodes:
            if n.op_type != "Relu":
                continue
            src = producer.get(n.input[0])
            while src is not None and src.op_type in ("Add", "Identity", "Cast"):
                src = producer.get(src.input[0])
            kinds.append("conv" if src is not None and src.op_type == "Conv" else "fc")
        return kinds

    def forward_ad(self, x: torch.Tensor, mask: torch.Tensor, tau: float | None = None):
        """AD pass. ``mask`` is (B,1,H,W) or (1,H,W) with 1 = visible.

        Returns the logits and, for every hidden ReLU layer in encoder order, a
        boolean tensor (B, n_units) that is True where the unit is still alive.
        """
        tau = self.tau if tau is None else tau
        m = mask.to(x.dtype)
        if m.dim() == 3:
            m = m.unsqueeze(1)
        if m.shape[1] != 1:
            m = (m > 0).any(dim=1, keepdim=True).to(x.dtype)
        if m.shape[0] == 1 and x.shape[0] > 1:
            m = m.expand(x.shape[0], -1, -1, -1)
        if x.shape[0] == 1 and m.shape[0] > 1:
            x = x.expand(m.shape[0], *x.shape[1:])

        z = x * m
        values = {self.input_name: z}
        masks = {self.input_name: m}
        alive = []
        self.relu_activations = []

        for node in self.nodes:
            attrs = self._attrs(node)
            ins = [values[name] if name in values else self._init(name) for name in node.input]
            out = self._run(node.op_type, ins, attrs)
            m_in = masks.get(node.input[0]) if node.input else None
            op = node.op_type

            if op == "Conv":
                w = ins[1]
                kh, kw = w.shape[-2:]
                pads = list(attrs.get("pads", [0, 0, 0, 0]))
                ones = torch.ones(1, 1, kh, kw, dtype=m_in.dtype) / (kh * kw)
                frac = F.conv2d(F.pad(m_in, [pads[1], pads[3], pads[0], pads[2]]), ones,
                                stride=list(attrs.get("strides", [1, 1])),
                                dilation=list(attrs.get("dilations", [1, 1])))
                m_out = (frac > tau).to(out.dtype)
                out = out * m_out
            elif op in ("MaxPool", "AveragePool"):
                k = list(attrs["kernel_shape"])
                s = list(attrs.get("strides", k))
                pads = list(attrs.get("pads", [0, 0, 0, 0]))
                frac = F.avg_pool2d(F.pad(m_in, [pads[1], pads[3], pads[0], pads[2]]),
                                    kernel_size=k, stride=s)
                m_out = (frac > tau).to(out.dtype)
                out = out * m_out
            elif op in ("Flatten", "Reshape") and m_in is not None and m_in.dim() == 4:
                full = m_in.expand(-1, values[node.input[0]].shape[1], -1, -1)
                m_out = full.flatten(1).mean(dim=1, keepdim=True)
            elif op in ("Gemm", "MatMul"):
                gate = (m_in > tau).to(out.dtype) if m_in.dim() == 2 else \
                    (m_in.flatten(1).mean(1, keepdim=True) > tau).to(out.dtype)
                out = out * gate
                m_out = gate
            elif op == "Relu":
                m_out = m_in
                full = m_in.expand_as(out) if m_in.dim() == 4 else m_in.expand(-1, out.shape[1])
                alive.append(full.reshape(out.shape[0], -1) > 0)
            else:
                m_out = m_in

            values[node.output[0]] = out
            masks[node.output[0]] = m_out

        return values[self.output_name], alive


def selftest(onnx_path, n: int = 8, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = ADOnnxModel(onnx_path)
    x = torch.rand(n, *model.input_shape[1:])
    with torch.no_grad():
        plain = model(x)
        plain_acts = [a.clone() for a in model.relu_activations]
        ones = torch.ones(n, 1, *model.input_shape[2:])
        ad, alive = model.forward_ad(x, ones)
        ad_acts = model.relu_activations
        # a fully masked image must kill every conv unit
        _, dead = model.forward_ad(x, torch.zeros_like(ones))
    return {
        "thm41_max_abs_diff": (plain - ad).abs().max().item(),
        "acts_equal": all(torch.equal(a, b) for a, b in zip(plain_acts, ad_acts)),
        "all_alive_with_ones": all(bool(a.all()) for a in alive),
        "all_dead_with_zeros": all(not bool(d.any()) for d in dead),
        "sizes": model.hidden_sizes,
        "kinds": model.hidden_kinds,
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = selftest(p)
        ok = r["thm41_max_abs_diff"] == 0.0 and r["acts_equal"] and r["all_alive_with_ones"] \
            and r["all_dead_with_zeros"]
        print(f"{Path(p).stem:6s} {'OK' if ok else 'FAIL'} {r}")
