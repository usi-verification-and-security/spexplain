"""A minimal, differentiable ONNX -> PyTorch interpreter.

It supports exactly the operator set spexplain's own C++ ONNX path
(`OnnxParser` / `Network2` / `OpenSMTVerifier2`, see the top-level CLAUDE.md)
understands: Gemm/MatMul/Add (fully-connected), Relu/Sigmoid, Conv, MaxPool,
AveragePool,
Flatten, Reshape, Transpose, plus the no-op/constant-folding nodes exporters
tend to add (Cast, Identity, Constant).

Why this exists: `--input-fix-sample-neuron-activations` files must number
hidden ReLU layers, and neurons within each layer, exactly the way
`OpenSMTVerifier2` does -- the k-th `Relu` node encountered while walking the
graph is hidden layer k, and its neurons are numbered in the tensor's natural
flatten order (row-major, i.e. channel-major for a conv output, matching
`CNNLayer::computeLayerOutput`, src/spexplain/network/CNNLayer.cpp). Loading
weights straight from the .onnx file -- rather than reconstructing a
per-architecture PyTorch `nn.Module` from a `.pth` checkpoint -- means one
code path works for *any* model spexplain can load, FC or CNN, without
needing to know its architecture ahead of time, and guarantees the ReLU
order/activation values seen here match what the SMT encoder sees (same
weights, same graph order).
"""

from __future__ import annotations

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper


def _as_torch(array):
    """Load a numpy array as a torch tensor, keeping integer tensors integral
    (shapes/perms/Cast targets) and floating everything else."""
    t = torch.from_numpy(array.copy())
    if t.dtype in (torch.float64, torch.float16):
        t = t.float()
    return t


class OnnxTorchModel(nn.Module):
    """Executes an ONNX graph with torch ops, so autograd (and hence PGD)
    works, and records post-ReLU activations in graph order.

    Only single-input, single-output graphs are supported -- true of every
    model in this repo and of everything `explain-onnx` can consume.
    """

    _SUPPORTED_OPS = {
        "Gemm", "MatMul", "Add", "Relu", "Sigmoid", "Conv", "MaxPool", "AveragePool",
        "Flatten", "Reshape", "Transpose", "Cast", "Identity", "Constant",
    }

    def __init__(self, onnx_path: str):
        super().__init__()
        model = onnx.load(str(onnx_path))
        graph = model.graph
        self.nodes = list(graph.node)

        unsupported = {n.op_type for n in self.nodes} - self._SUPPORTED_OPS
        if unsupported:
            raise NotImplementedError(
                f"{onnx_path}: unsupported ONNX op(s) for this interpreter: "
                f"{sorted(unsupported)} (extend OnnxTorchModel to add them; "
                f"keep it in sync with src/spexplain/network/OnnxParser.cpp)"
            )

        self._buffer_names: dict[str, str] = {}
        for i, init in enumerate(graph.initializer):
            buf_name = f"init_{i}"
            self.register_buffer(buf_name, _as_torch(numpy_helper.to_array(init)))
            self._buffer_names[init.name] = buf_name

        if len(graph.input) != 1:
            raise ValueError(f"{onnx_path}: expected a single graph input, got {len(graph.input)}")
        if len(graph.output) != 1:
            raise ValueError(f"{onnx_path}: expected a single graph output, got {len(graph.output)}")
        self.input_name = graph.input[0].name
        self.output_name = graph.output[0].name
        self.input_shape = [d.dim_value if d.dim_value else 1
                             for d in graph.input[0].type.tensor_type.shape.dim]
        self.output_size = graph.output[0].type.tensor_type.shape.dim[-1].dim_value

        # ReLU activations captured by the most recent forward() call, in the
        # order OpenSMTVerifier2 assigns its ReLU-layer indices.
        self.relu_activations: list[torch.Tensor] = []
        self.n_relu_layers = sum(1 for n in self.nodes if n.op_type == "Relu")

    def _init(self, name: str) -> torch.Tensor:
        return getattr(self, self._buffer_names[name])

    @staticmethod
    def _attrs(node) -> dict:
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values: dict[str, torch.Tensor] = {self.input_name: x}
        self.relu_activations = []

        for node in self.nodes:
            attrs = self._attrs(node)
            ins = [values[name] if name in values else self._init(name) for name in node.input]
            out = self._run(node.op_type, ins, attrs)
            values[node.output[0]] = out

        return values[self.output_name]

    def _run(self, op: str, ins: list, attrs: dict) -> torch.Tensor:
        if op in ("Gemm", "MatMul"):
            a, b = ins[0], ins[1]
            if attrs.get("transA"):
                a = a.transpose(-2, -1)
            if attrs.get("transB"):
                b = b.transpose(-2, -1)
            out = a.matmul(b)
            alpha = attrs.get("alpha", 1.0)
            if alpha != 1.0:
                out = out * alpha
            if len(ins) > 2:
                beta = attrs.get("beta", 1.0)
                out = out + (ins[2] * beta if beta != 1.0 else ins[2])
            return out

        if op == "Add":
            return ins[0] + ins[1]

        if op == "Relu":
            out = F.relu(ins[0])
            self.relu_activations.append(out)
            return out

        if op == "Sigmoid":
            return torch.sigmoid(ins[0])

        if op == "Conv":
            weight = ins[1]
            bias = ins[2] if len(ins) > 2 else None
            kh, kw = weight.shape[-2:]
            pads = list(attrs.get("pads", [0, 0, 0, 0]))
            strides = list(attrs.get("strides", [1, 1]))
            dilations = list(attrs.get("dilations", [1, 1]))
            group = attrs.get("group", 1)
            x_in = ins[0]
            if pads[:2] != pads[2:]:  # asymmetric: pad explicitly, then conv with padding=0
                x_in = F.pad(x_in, [pads[1], pads[3], pads[0], pads[2]])
                pads = [0, 0]
            else:
                pads = pads[:2]
            return F.conv2d(x_in, weight, bias, stride=strides, padding=pads,
                             dilation=dilations, groups=group)

        if op == "MaxPool":
            kernel_shape = list(attrs["kernel_shape"])
            strides = list(attrs.get("strides", kernel_shape))
            pads = list(attrs.get("pads", [0] * 2 * len(kernel_shape)))
            x_in = ins[0]
            if pads[:2] != pads[2:]:  # asymmetric: pad with -inf explicitly (identity for max)
                x_in = F.pad(x_in, [pads[1], pads[3], pads[0], pads[2]], value=float("-inf"))
                pads = [0, 0]
            else:
                pads = pads[:2]
            return F.max_pool2d(x_in, kernel_size=kernel_shape, stride=strides, padding=pads)

        if op == "AveragePool":
            kernel_shape = list(attrs["kernel_shape"])
            strides = list(attrs.get("strides", kernel_shape))
            pads = list(attrs.get("pads", [0] * 2 * len(kernel_shape)))
            count_include_pad = bool(attrs.get("count_include_pad", 0))
            x_in = ins[0]
            if pads[:2] != pads[2:]:
                # Asymmetric: pad with zeros explicitly. Only exact when the padding is counted,
                # which is what count_include_pad=1 asks for.
                if not count_include_pad and any(pads):
                    raise NotImplementedError(
                        "AveragePool with asymmetric pads and count_include_pad=0")
                x_in = F.pad(x_in, [pads[1], pads[3], pads[0], pads[2]])
                pads = [0, 0]
            else:
                pads = pads[:2]
            return F.avg_pool2d(x_in, kernel_size=kernel_shape, stride=strides, padding=pads,
                                count_include_pad=count_include_pad)

        if op == "Flatten":
            return ins[0].flatten(attrs.get("axis", 1))

        if op == "Reshape":
            shape = ins[1].long().tolist()
            shape = [d if d != 0 else ins[0].shape[i] for i, d in enumerate(shape)]
            return ins[0].reshape(shape)

        if op == "Transpose":
            perm = list(attrs.get("perm", list(reversed(range(ins[0].dim())))))
            return ins[0].permute(perm)

        if op in ("Cast", "Identity"):
            to = attrs.get("to")
            if to == onnx.TensorProto.INT64:
                return ins[0].long()
            return ins[0].float()

        if op == "Constant":
            return _as_torch(numpy_helper.to_array(attrs["value"]))

        raise NotImplementedError(f"Unsupported ONNX op for this interpreter: {op}")
