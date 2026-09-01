#!/usr/bin/env python3
"""Convert PyTorch ``.pth`` checkpoints of fully-connected classifiers to ONNX.

The checkpoints produced for this project store a plain ``nn.Sequential`` of
``Linear`` layers separated by activations, e.g.::

    {'model_state_dict': OrderedDict([('net.0.weight', ...), ('net.0.bias', ...),
                                      ('net.2.weight', ...), ...]),
     'input_dim': 13, 'hidden_size': 50, 'num_layers': 1, 'output_dim': 1}

The architecture is reconstructed from the *state dict itself* (module indices
and tensor shapes), so checkpoints without the metadata keys convert as well;
when the metadata is present it is cross-checked and a mismatch is an error.

By default no trailing activation is appended, so the ONNX output is the raw
logit -- exactly matching the ``.nnet`` semantics and what spexplain expects
(see ``--drop-sigmoid`` in README.md). Use ``--final-activation sigmoid`` to
append one anyway; spexplain drops it again on load.

Examples::

    ./convert_pth_to_onnx.py ../data/models/heart_attack/heart_attack_50x1.pth
    ./convert_pth_to_onnx.py ../data/models/heart_attack --pattern '*.pth' --verify
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STATE_DICT_KEYS = ("model_state_dict", "state_dict", "net", "model")

ACTIVATIONS = ("relu", "sigmoid", "tanh", "none")

_PARAM_RE = re.compile(r"^(?P<prefix>.*?)(?P<idx>\d+)\.(?P<kind>weight|bias)$")


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Missing dependency for PTH->ONNX conversion. Install with: "
            "python3 -m pip install torch onnx"
        ) from exc
    import torch

    return torch


def _make_activation(name: str):
    _require_torch()
    from torch import nn

    return {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }[name]()


def extract_state_dict(obj):
    """Pull the parameter mapping out of whatever ``torch.load`` returned."""
    torch = _require_torch()
    from torch import nn

    if isinstance(obj, nn.Module):
        return None, obj

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint payload of type {type(obj).__name__}")

    # A checkpoint dict wrapping the parameters under a well-known key.
    for key in STATE_DICT_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict) and inner:
            return inner, None

    # Or the bare state dict itself.
    if all(hasattr(v, "shape") for v in obj.values()):
        return obj, None

    raise ValueError(
        f"Could not locate a state dict in checkpoint; top-level keys: {sorted(obj)}"
    )


def infer_layout(indices: list[int]) -> str:
    """Decide whether the module indices are Sequential positions or plain ordinals.

    ``nn.Sequential`` numbers *every* child, so activations show up as gaps
    (``net.0``, ``net.2``, ``net.4``). Named attributes (``layer1``, ``layer2``,
    ``layer3``) are numbered consecutively and their activations live in
    ``forward()``, invisible to the state dict. Getting this wrong is silent:
    a 'named' checkpoint read as 'sequential' yields a net with no activations
    at all, which is just one big linear map.
    """
    if len(indices) < 2:
        return "sequential"
    gaps = {b - a for a, b in zip(indices, indices[1:])}
    return "named" if gaps == {1} else "sequential"


def build_model(state_dict, activation: str, final_activation: str, layout: str):
    """Rebuild the network that ``state_dict`` was saved from.

    For the ``sequential`` layout the Linear layers are placed at their original
    module indices and the gaps are filled with activations, so
    ``load_state_dict(strict=True)`` succeeds with the untouched key names --
    which is what makes the reconstruction exact. Because the gaps are read per
    layer, a net carrying activations on only *some* layers (``Linear, ReLU,
    Linear, Linear`` -> indices 0, 2, 3) reconstructs correctly with no flags.

    ``--activation none`` is therefore only meaningful for a gapless net, where
    it produces a Linear-only stack; in the ``sequential`` layout a gap with
    nothing to put in it is an error rather than a silently dropped module.
    """
    torch = _require_torch()
    from torch import nn

    weight_names = []  # module names, in state-dict (registration) order
    layers = {}  # module name -> (out_features, in_features)
    for key, tensor in state_dict.items():
        if key.endswith(".weight"):
            if tensor.dim() != 2:
                raise ValueError(
                    f"Only fully-connected models are supported; {key} has shape {tuple(tensor.shape)}"
                )
            name = key[: -len(".weight")]
            weight_names.append(name)
            layers[name] = tuple(tensor.shape)
        elif not key.endswith(".bias"):
            raise ValueError(f"Unexpected parameter (neither weight nor bias): {key}")

    if not layers:
        raise ValueError("No Linear layers found in the state dict")

    for name in weight_names:
        if f"{name}.bias" not in state_dict:
            raise ValueError(f"Linear layer {name} has a weight but no bias")

    # nn.Sequential names its children by position, so activations show up as
    # index gaps. Any other naming (layer1/layer2/output_layer) carries no
    # position information, so fall back to registration order.
    matches = [_PARAM_RE.match(f"{name}.weight") for name in weight_names]
    prefixes = {m.group("prefix") for m in matches} if all(matches) else None
    if prefixes is not None and len(prefixes) == 1:
        prefix = prefixes.pop()
        order = sorted(weight_names, key=lambda n: int(n[len(prefix) :]))
        indices = [int(n[len(prefix) :]) for n in order]
    else:
        prefix = None
        order = weight_names
        indices = list(range(len(order)))

    # Check the layers actually chain: out_features of one == in_features of the next.
    for prev, cur in zip(order, order[1:]):
        if layers[prev][0] != layers[cur][1]:
            raise ValueError(
                f"Layer shapes do not chain: {prev} outputs {layers[prev][0]} "
                f"but {cur} expects {layers[cur][1]}"
            )

    if layout == "auto":
        layout = "named" if prefix is None else infer_layout(indices)

    slot_of = dict(zip(indices, order))
    modules = []
    if layout == "named":
        # No positional information: put an activation between every pair of Linears.
        for position, name in enumerate(order):
            out_features, in_features = layers[name]
            if position and activation != "none":
                modules.append(_make_activation(activation))
            modules.append(nn.Linear(in_features, out_features))
    else:
        for slot in range(indices[-1] + 1):
            if slot in slot_of:
                out_features, in_features = layers[slot_of[slot]]
                modules.append(nn.Linear(in_features, out_features))
            elif activation == "none":
                raise ValueError(
                    f"Module index {slot} is a gap between Linear layers, so some "
                    "module was registered there, but --activation none leaves "
                    "nothing to put in its place. Note that a Sequential with "
                    "activations on only some layers needs no flag: the gaps "
                    "already say which layers have one."
                )
            else:
                modules.append(_make_activation(activation))

    if final_activation != "none":
        modules.append(_make_activation(final_activation))
    net = nn.Sequential(*modules)

    if layout == "named":
        # The rebuilt Sequential has different key names, so copy tensors across
        # in layer order and check every shape as we go.
        linears = [m for m in net if isinstance(m, nn.Linear)]
        assert len(linears) == len(order)
        with torch.no_grad():
            for linear, name in zip(linears, order):
                weight = state_dict[f"{name}.weight"]
                bias = state_dict[f"{name}.bias"]
                if tuple(weight.shape) != tuple(linear.weight.shape):
                    raise ValueError(f"Shape mismatch for {name}.weight")
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
    elif prefix:
        # Load under the checkpoint's own prefix so strict=True validates every key.
        holder = nn.Module()
        setattr(holder, prefix.rstrip("."), net)
        holder.load_state_dict(state_dict, strict=True)
    else:
        net.load_state_dict(state_dict, strict=True)

    net.eval()

    in_features = layers[order[0]][1]
    out_features = layers[order[-1]][0]
    return net, in_features, out_features, layout


def check_metadata(checkpoint, in_features: int, out_features: int, n_linear: int) -> None:
    """Cross-check the inferred shape against the checkpoint's own metadata."""
    if not isinstance(checkpoint, dict):
        return

    expected = {
        "input_dim": in_features,
        # Binary checkpoints store output_dim, multi-class ones store num_classes.
        "output_dim": out_features,
        "num_classes": out_features,
        # num_layers counts hidden layers; the output Linear is the extra one.
        "num_layers": n_linear - 1,
    }
    for key, inferred in expected.items():
        stored = checkpoint.get(key)
        if stored is not None and int(stored) != inferred:
            raise ValueError(
                f"Checkpoint metadata {key}={stored} contradicts the state dict ({inferred})"
            )


def load_dataset(csv_path: Path, in_features: int, samples: int):
    """Read the leading feature columns of a spexplain dataset CSV."""
    import numpy as np

    rows = []
    with csv_path.open() as handle:
        header = handle.readline()  # feature names, e.g. 'age,sex,...,output'
        if header.split(",")[0].replace(".", "").replace("-", "").isdigit():
            handle.seek(0)  # headerless file: keep the first line
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split(",")[:in_features]]
            if len(values) != in_features:
                raise ValueError(
                    f"{csv_path}: expected at least {in_features} columns, got {len(values)}"
                )
            rows.append(values)
            if samples and len(rows) >= samples:
                break

    if not rows:
        raise ValueError(f"{csv_path}: no data rows")
    return np.asarray(rows, dtype=np.float32)


def verify(
    model, onnx_path: Path, in_features: int, samples: int, tol: float, dataset: Path | None
) -> tuple[float, float]:
    """Run inputs through torch and onnxruntime; return (max abs, max rel) diff."""
    torch = _require_torch()

    import numpy as np

    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "--verify needs onnxruntime: python3 -m pip install onnxruntime"
        ) from exc

    if dataset is not None:
        inputs = load_dataset(dataset, in_features, samples)
    else:
        # A unit-range probe is only a smoke test -- some of these models take raw
        # unnormalised features -- but any structural error (transposed weights,
        # dropped activation, reordered layers) shows up as an O(1) difference at
        # any input scale. Pass --dataset to check at the real input magnitudes.
        rng = np.random.default_rng(0)
        inputs = rng.random((samples, in_features), dtype=np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(inputs)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    spec = session.get_inputs()[0]
    batch_dim = spec.shape[0] if spec.shape else None
    if isinstance(batch_dim, int) and batch_dim != len(inputs):
        # Model was exported with a fixed batch size: feed the rows one by one.
        rows = [session.run(None, {spec.name: row[None, :]})[0] for row in inputs]
        onnx_out = np.concatenate(rows, axis=0)
    else:
        onnx_out = session.run(None, {spec.name: inputs})[0]

    abs_diff = np.abs(torch_out - onnx_out)
    # Deep nets produce logits in the hundreds, where float32 accumulation alone
    # costs ~1e-5 absolutely; judge on relative error, with a floor so that
    # near-zero outputs do not blow the ratio up.
    scale = np.maximum(np.maximum(np.abs(torch_out), np.abs(onnx_out)), 1.0)
    rel_diff = float(np.max(abs_diff / scale))
    if rel_diff > tol:
        raise ValueError(
            f"torch and onnxruntime disagree: max rel diff {rel_diff:.3g} > tol {tol:.3g} "
            f"(max abs diff {float(np.max(abs_diff)):.3g})"
        )
    return float(np.max(abs_diff)), rel_diff


def convert(
    pth_path: Path,
    *,
    out_path: Path | None,
    overwrite: bool,
    opset: int,
    activation: str,
    final_activation: str,
    dynamic_batch: bool,
    do_verify: bool,
    samples: int,
    tol: float,
    dataset: Path | None,
    layout: str,
) -> tuple[Path, str]:
    torch = _require_torch()

    onnx_path = out_path or pth_path.with_suffix(".onnx")
    if onnx_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {onnx_path}")

    checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    state_dict, module = extract_state_dict(checkpoint)

    if module is not None:
        model = module.eval()
        first = next(p for p in model.parameters() if p.dim() == 2)
        in_features = first.shape[1]
        out_features = None
        n_linear = None
        layout = "module"
    else:
        model, in_features, out_features, layout = build_model(
            state_dict, activation, final_activation, layout
        )
        n_linear = sum(1 for k in state_dict if k.endswith(".weight"))
        check_metadata(checkpoint, in_features, out_features, n_linear)

    dummy = torch.zeros(1, in_features, dtype=torch.float32)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        # The TorchScript exporter emits the plain Gemm/Relu graph that
        # spexplain's OnnxParser understands; the dynamo exporter does not.
        dynamo=False,
    )

    note = f"{in_features} -> {out_features}" if out_features else f"{in_features} inputs"
    if n_linear:
        note += f", {n_linear} Linear, {layout}"
    if final_activation != "none":
        note += f", +{final_activation}"

    if do_verify:
        try:
            abs_diff, rel_diff = verify(model, onnx_path, in_features, samples, tol, dataset)
        except Exception:
            # Never leave an unverified file behind under a FAIL line.
            onnx_path.unlink(missing_ok=True)
            raise
        note += f", max|diff|={abs_diff:.2e} rel={rel_diff:.2e}"

    return onnx_path, note


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth checkpoints of FC classifiers to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="A .pth file, or a directory to scan")
    parser.add_argument("-o", "--output", type=Path, help="Output .onnx path (single-file mode only)")
    parser.add_argument("--pattern", default="*.pth", help="Glob used when PATH is a directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .onnx files")
    parser.add_argument("--opset", type=int, default=15, help="ONNX opset version")
    parser.add_argument(
        "--activation",
        choices=ACTIVATIONS,
        default="relu",
        help="Activation used where the checkpoint implies one (state dicts do "
        "not record the type). 'none' only applies to gapless checkpoints",
    )
    parser.add_argument(
        "--final-activation",
        choices=ACTIVATIONS,
        default="none",
        help="Activation appended after the output layer ('none' keeps the raw logit)",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "sequential", "named"),
        default="auto",
        help="'sequential' = nn.Sequential indices with activations in the gaps "
        "(net.0/net.2/...); 'named' = consecutively numbered attributes "
        "(layer1/layer2/...) with an activation between every pair",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with a dynamic batch axis instead of a fixed batch of 1",
    )
    parser.add_argument("--verify", action="store_true", help="Compare torch vs onnxruntime outputs")
    parser.add_argument(
        "--dataset", type=Path, help="CSV of real inputs for --verify (default: random probes)"
    )
    parser.add_argument("--samples", type=int, default=64, help="Number of samples used by --verify")
    parser.add_argument(
        "--tol", type=float, default=1e-4, help="Relative tolerance used by --verify"
    )
    args = parser.parse_args()

    if args.path.is_dir():
        pth_files = sorted(args.path.glob(args.pattern))
        if args.output:
            print("Error: --output is only valid for a single input file", file=sys.stderr)
            return 2
        if not pth_files:
            print(f"No checkpoints matched '{args.pattern}' in {args.path}")
            return 0
    elif args.path.is_file():
        pth_files = [args.path]
    else:
        print(f"Error: no such file or directory: {args.path}", file=sys.stderr)
        return 2

    print(f"Converting {len(pth_files)} checkpoint(s)")

    failures = []
    for pth_path in pth_files:
        try:
            onnx_path, note = convert(
                pth_path,
                out_path=args.output,
                overwrite=args.overwrite,
                opset=args.opset,
                activation=args.activation,
                final_activation=args.final_activation,
                dynamic_batch=args.dynamic_batch,
                do_verify=args.verify,
                samples=args.samples,
                tol=args.tol,
                dataset=args.dataset,
                layout=args.layout,
            )
            print(f"OK   {pth_path.name} -> {onnx_path.name}  ({note})")
        except Exception as exc:
            failures.append((pth_path, str(exc)))
            print(f"FAIL {pth_path.name}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} conversion(s) failed.", file=sys.stderr)
        return 1

    print("\nAll conversions completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
