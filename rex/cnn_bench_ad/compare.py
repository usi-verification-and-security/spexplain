#!/usr/bin/env python3
"""Tables for the report, and the overlap between the two neuron heuristics.

Heuristic ADV (adversarial search, ``activation_diff.py``): for each sample,
the ReLUs whose phase did NOT flip between the sample and a PGD counterfactual
-- the set spexplain's hybrid mode fixes.

Heuristic AD (ReX + ConvAD, ``neuron_files.py``): the deactivated ReLUs, whose
receptive field lies entirely outside the ReX explanation; and, for contrast,
the "inside" ReLUs, whose receptive field lies entirely inside it.

Both are read back with a re-implementation of spexplain's parser
(``parseSampleNeuronActivationsImpl`` in src/spexplain/framework/Config.cpp),
which also validates them. Overlap is pooled over (sample, neuron) pairs of
the samples present in both files.

Writes ``out/tables.md`` and ``out/overlap_per_layer.csv``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODELS = REPO / "data/models/gtsrb/cnn-bench"
ADV_DIR = REPO / "data/neuron_activations/gtsrb/cnn-bench"
NA = HERE / "neuron_activations"
OUT = HERE / "out"
FAMILIES = {"NS": [f"NS{i}" for i in range(1, 7)], "NAB": [f"NAB{i}" for i in range(1, 7)]}
GAMMA = "0.5"


def parse(path: Path, sizes: list[int]) -> dict[int, list[set[int]]]:
    """spexplain's reader: sample index, then per hidden layer indices up to 0."""
    toks = path.read_text().split()
    pos, out = 0, {}
    while pos < len(toks):
        sid = int(toks[pos]); pos += 1
        if sid < 1 or sid in out:
            raise ValueError(f"{path.name}: bad or repeated sample index {sid}")
        layers = []
        for li, size in enumerate(sizes):
            cur = set()
            while True:
                v = int(toks[pos]); pos += 1
                if v == 0:
                    break
                if not 1 <= v <= size:
                    raise ValueError(f"{path.name}: sample {sid} layer {li+1}: {v} not in 1..{size}")
                cur.add(v - 1)
            layers.append(cur)
        out[sid] = layers
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    manifest = pd.read_csv(MODELS / "manifest.csv").set_index("id")
    t_models, t_expl, t_files, t_depth, t_adv_depth = [], [], [], [], []
    t_overlap, t_overlap_depth, rows_csv = [], [], []
    max_layers = 0

    for fam, names in FAMILIES.items():
        for m in names:
            meta = json.loads((NA / f"{m}_ad_meta.json").read_text())
            sizes, kinds = meta["hidden_sizes"], meta["hidden_kinds"]
            max_layers = max(max_layers, len(sizes))
            r = manifest.loc[m]
            t_models.append([m, f"`{r.layers}`", f"{r.relu_neurons:,}", f"{r.params:,}",
                             f"{r.test_acc:.3f}", f"{r.onnx_rel_diff}"])

            sz = pd.read_csv(OUT / f"{m}_ad_g{GAMMA}_sizes.csv")
            ok = sz[sz.status == "ok"]
            t_expl.append([m, f"{len(ok)}/{len(sz)}",
                           f"{ok.area_px.mean():.1f}px ({ok.area_px.mean() / 1024:.1%})",
                           f"{int(ok.area_px.min())}–{int(ok.area_px.max())}",
                           f"{ok.conf.mean():.3f}", f"{sz.secs.sum() / 60:.1f} min"])

            deact = parse(NA / f"{m}_ad_deactives.txt", sizes)
            act = parse(NA / f"{m}_ad_actives.txt", sizes)
            inside = parse(NA / f"{m}_ad_inside.txt", sizes)
            for sid in deact:  # actives/deactives must partition every layer
                for li, n in enumerate(sizes):
                    if deact[sid][li] & act[sid][li] or len(deact[sid][li]) + len(act[sid][li]) != n:
                        raise ValueError(f"{m} sample {sid} layer {li+1}: not a partition")
            t_files.append([m, len(sizes), f"{sizes}", pct(meta["mean_deactive_frac_total"]),
                            pct(meta["mean_inside_frac_total"]), "yes"])
            per = meta["mean_deactive_frac_per_layer"]
            t_depth.append([m] + [f"{100*f:.0f}%" + (" (FC)" if k == "fc" else "")
                                  for f, k in zip(per, kinds)])

            adv = parse(ADV_DIR / f"{m}.txt", sizes)
            common = sorted(set(adv) & set(deact))
            stable_frac = [sum(len(adv[s][li]) for s in adv) / (len(adv) * n)
                           for li, n in enumerate(sizes)]
            t_adv_depth.append([m] + [f"{100*f:.0f}%" + (" (FC)" if k == "fc" else "")
                                      for f, k in zip(stable_frac, kinds)])

            tot = {k: 0 for k in ("n", "S", "D", "I", "SD", "SI", "conv_n", "conv_S", "conv_D",
                                  "conv_SD", "conv_I", "conv_SI")}
            depth_cells = []
            for li, n in enumerate(sizes):
                c = {"n": 0, "S": 0, "D": 0, "I": 0, "SD": 0, "SI": 0}
                for s in common:
                    S, D, I = adv[s][li], deact[s][li], inside[s][li]
                    c["n"] += n; c["S"] += len(S); c["D"] += len(D); c["I"] += len(I)
                    c["SD"] += len(S & D); c["SI"] += len(S & I)
                for k in c:
                    tot[k] += c[k]
                    if kinds[li] == "conv" and f"conv_{k}" in tot:
                        tot[f"conv_{k}"] += c[k]
                p_s = c["S"] / c["n"]
                p_s_d = c["SD"] / c["D"] if c["D"] else float("nan")
                p_s_nd = (c["S"] - c["SD"]) / (c["n"] - c["D"])
                p_s_i = c["SI"] / c["I"] if c["I"] else float("nan")
                flipped = c["n"] - c["S"]
                rows_csv.append({
                    "model": m, "layer": li + 1, "kind": kinds[li], "units": n,
                    "samples": len(common), "stable_frac": p_s, "deact_frac": c["D"] / c["n"],
                    "inside_frac": c["I"] / c["n"],
                    "P_stable_given_deact": p_s_d, "P_stable_given_not_deact": p_s_nd,
                    "P_stable_given_inside": p_s_i,
                    "jaccard_S_D": c["SD"] / (c["S"] + c["D"] - c["SD"]),
                    "adv_covered_by_deact": c["SD"] / c["S"] if c["S"] else float("nan"),
                    "flipped_in_deact": (c["D"] - c["SD"]) / flipped if flipped else float("nan"),
                    "flipped_in_inside": (c["I"] - c["SI"]) / flipped if flipped else float("nan"),
                })
                f = lambda v: "–" if v != v else f"{100*v:.0f}%"  # noqa: E731
                depth_cells.append(f"{f(p_s)} / {f(p_s_d)} / {f(p_s_nd)}"
                                   + (" (FC)" if kinds[li] == "fc" else ""))
            t_overlap_depth.append([m] + depth_cells)

            def ov(pre):
                n, S, D = tot[f"{pre}n"], tot[f"{pre}S"], tot[f"{pre}D"]
                SD, I, SI = tot[f"{pre}SD"], tot[f"{pre}I"], tot[f"{pre}SI"]
                return [pct(S / n), pct(D / n), pct(SD / n), pct((S * D) / (n * n)),
                        pct(SD / (S + D - SD)), pct(SD / D), pct((S - SD) / (n - D)),
                        pct(SD / S), pct(SI / I) if I else "–",
                        pct((D - SD) / (n - S)) if n > S else "–"]
            t_overlap.append([m, len(common)] + ov("conv_"))

    def table(header, rows):
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(lines)

    def pad(rows, width):
        return [r + ["–"] * (width + 1 - len(r)) for r in rows]

    layer_hdr = [f"layer {i}" for i in range(1, max_layers + 1)]
    md = [
        "## Models", table(["model", "architecture", "ReLU", "params", "test acc", "ONNX diff"],
                           t_models),
        f"## Explanations (ReX + ConvAD, γ={GAMMA})",
        table(["model", "explained", "mean size", "range", "mean conf", "time"], t_expl),
        "## Neuron files",
        table(["model", "layers", "sizes", "% deactive", "% inside", "spexplain-compatible"],
              t_files),
        "## Deactivation per depth (ReX + ConvAD)",
        table(["model"] + layer_hdr, pad(t_depth, max_layers)),
        "## Fixed (phase-stable) per depth (adversarial search)",
        table(["model"] + layer_hdr, pad(t_adv_depth, max_layers)),
        "## Overlap, conv layers pooled",
        table(["model", "samples", "ADV fixed", "AD deact", "both", "both if independent",
               "Jaccard", "P(fixed | deact)", "P(fixed | not deact)", "ADV covered by deact",
               "P(fixed | inside)", "flipped that are deact"], t_overlap),
        "## Per depth: P(fixed) / P(fixed | deact) / P(fixed | not deact)",
        table(["model"] + layer_hdr, pad(t_overlap_depth, max_layers)),
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "tables.md").write_text("\n\n".join(md) + "\n")
    with (OUT / "overlap_per_layer.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
