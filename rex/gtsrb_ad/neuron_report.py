#!/usr/bin/env python3
"""Record which neurons ConvAD switches ON and OFF, per sample, per hidden layer.

A neuron is **deactive** when the AD mask is 0 at its position -- i.e. its whole
receptive field lies outside the ReX explanation mask, so AD has switched that
part of the model off. This is the A/D notion, not the ReLU sign.

Output format
-------------
Matches spexplain's ``--input-fix-sample-neuron-activations`` parser
(``Config.cpp::parseSampleNeuronActivationsImpl``), which reads::

    while (ifs >> sampleIdx)
        for layerIdx in 1 .. nHiddenLayers:
            read node indices until a 0 is seen

so each record is::

    <sample index>                       (1-based)
    <node> <node> ... 0                  hidden layer 1
    <node> <node> ... 0                  hidden layer 2
    ...                                  exactly nHiddenLayers lines

Both indices are 1-based (the parser does ``idx - 1`` on each, and rejects 0 as
a node because 0 is the terminator). A layer with nothing to report is written
as a bare ``0``, which the parser needs in order to stay in sync -- the line
count per sample must always equal the number of hidden layers.

Node order inside a layer is the flattened conv output in **NCHW row-major**
order, ``index = c*H*W + h*W + w`` (+1 for 1-based). That is what ``nn.Flatten``
and the ONNX ``Flatten`` spec both produce, and OnnxParser.cpp works in NCHW,
so it lines up with how spexplain numbers the same units.

Hidden layers are the ReLU layers, in forward order. For these models that is
one per conv plus one per hidden FC; the final Linear to 43 logits is the
output layer and is not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from interpretable_gtsrb import InterpretableGTSRB

HERE = Path(__file__).resolve().parent


def masks_for_sample(model, x, mask, keep=None):
    """Per-layer boolean tensors, True where the unit is ACTIVE.

    The forward pass returns one entry per unit-producing layer (Conv2d,
    MaxPool2d, Linear). ``keep`` selects which of those to return; by default
    the caller passes ``model.relu_mask_indices()`` so only ReLU-followed
    layers appear.
    """
    with torch.no_grad():
        _, active = model(
            x, explanation_mode=True, explanation_mask=mask, return_masks=True
        )
    if keep is not None:
        active = [active[i] for i in keep]
    return active


def _indices(flat_bool) -> list[int]:
    """1-based indices where flat_bool is True."""
    return (torch.nonzero(flat_bool, as_tuple=False).flatten() + 1).tolist()


def write_activation_file(path: Path, records, n_layers: int, blank_between: bool = True):
    """records: list of (sample_index_1based, [list-of-indices per layer]).

    No header or comments. spexplain's parser is a bare ``ifs >> long`` loop, so
    any non-numeric byte in the file makes it throw "Expected sample index".
    All metadata goes to the sidecar ``*_ad_meta.json`` instead.

    ``blank_between`` puts an empty line between samples for readability. This
    is safe: ``operator>>`` skips all whitespace, so blank lines are invisible
    to the parser.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for n, (sample_idx, per_layer) in enumerate(records):
            assert len(per_layer) == n_layers, (
                f"sample {sample_idx}: {len(per_layer)} layers, expected {n_layers}"
            )
            if blank_between and n:
                fh.write("\n")
            fh.write(f"{sample_idx}\n")
            for idxs in per_layer:
                fh.write(" ".join(str(i) for i in idxs) + (" 0\n" if idxs else "0\n"))


def collect(model, samples, masks, keep=None):
    """samples: (N,C,H,W); masks: (N,1,H,W) with 1 = visible.

    Returns (active_records, deactive_records).
    """
    act_recs, deact_recs = [], []
    for i in range(samples.shape[0]):
        x = samples[i : i + 1]
        m = masks[i : i + 1]
        per_layer = masks_for_sample(model, x, m, keep=keep)
        a_layers, d_layers = [], []
        for act in per_layer:
            flat = act[0].reshape(-1)
            a_layers.append(_indices(flat))
            d_layers.append(_indices(~flat))
        act_recs.append((i + 1, a_layers))
        deact_recs.append((i + 1, d_layers))
    return act_recs, deact_recs


def summarise(records, sizes):
    """Mean count per layer, and overall fraction."""
    n = len(records)
    per_layer = [0] * len(sizes)
    for _, layers in records:
        for j, idxs in enumerate(layers):
            per_layer[j] += len(idxs)
    means = [c / n for c in per_layer]
    total = sum(per_layer) / n
    return means, total, total / sum(sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name, e.g. gtsrb_c")
    ap.add_argument("--masks", required=True, type=Path,
                    help=".pt produced by run_rex.py: {'x':(N,C,H,W), 'mask':(N,1,H,W), ...}")
    ap.add_argument("--outdir", type=Path,
                    default=Path("/Users/labbaf/Documents/rex/neuron_activations"))
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--include-non-relu", action="store_true",
                    help="also record layers NOT followed by a ReLU (max-pool "
                         "stages and the output Linear). Off by default. NOTE: "
                         "turning this on makes the file unreadable by spexplain, "
                         "whose nHiddenLayers counts only ReLU layers, so the "
                         "extra lines would desynchronise its parser.")
    ap.add_argument("--no-blank-lines", action="store_true",
                    help="do not separate samples with a blank line")
    args = ap.parse_args()

    ckpt = HERE / "models" / f"{args.model}.pt"
    model = InterpretableGTSRB(ckpt, tau=args.tau)
    model.eval()

    blob = torch.load(args.masks, map_location="cpu", weights_only=False)
    samples, masks = blob["x"], blob["mask"]

    if args.include_non_relu:
        keep = list(range(len(model.unit_layers)))
        sizes = model.all_layer_sizes()
        layer_desc = [f"{d['kind']}{'' if d['followed_by_relu'] else ' (no ReLU)'}"
                      for d in model.unit_layers]
    else:
        keep = model.relu_mask_indices()
        sizes = model.hidden_layer_sizes()
        layer_desc = [d["kind"] for d in model.unit_layers if d["followed_by_relu"]]
    nL = len(sizes)

    act, deact = collect(model, samples, masks, keep=keep)

    suffix = "_all" if args.include_non_relu else ""
    a_path = args.outdir / f"{args.model}_ad_actives{suffix}.txt"
    d_path = args.outdir / f"{args.model}_ad_deactives{suffix}.txt"
    blank = not args.no_blank_lines
    write_activation_file(a_path, act, nL, blank_between=blank)
    write_activation_file(d_path, deact, nL, blank_between=blank)

    am, at, af = summarise(act, sizes)
    dm, dt, df = summarise(deact, sizes)
    meta = {
        "model": args.model,
        "arch": "-".join(model.tokens) + "-fc",
        "tau": args.tau,
        "include_non_relu": bool(args.include_non_relu),
        "layers_recorded": layer_desc,
        "spexplain_compatible": not args.include_non_relu,
        "n_hidden_layers": nL,
        "hidden_shapes": [list(s) for s in model.hidden_shapes],
        "hidden_sizes": sizes,
        "total_relu_neurons": sum(sizes),
        "n_samples": len(act),
        "mean_active_per_layer": am,
        "mean_deactive_per_layer": dm,
        "mean_active_total": at,
        "mean_deactive_total": dt,
        "frac_deactive": df,
        "files": {"actives": a_path.name, "deactives": d_path.name},
    }
    m_path = args.outdir / f"{args.model}_ad_meta{suffix}.json"
    m_path.write_text(json.dumps(meta, indent=2))

    print(f"{args.model}: {nL} hidden layers, sizes {sizes} (total {sum(sizes)})")
    print(f"  mean deactive/sample {dt:.1f} ({df:.1%})   mean active/sample {at:.1f}")
    print(f"  per-layer deactive means: {[round(v,1) for v in dm]}")
    print(f"  wrote {a_path.name}, {d_path.name}, {m_path.name}")


if __name__ == "__main__":
    main()
