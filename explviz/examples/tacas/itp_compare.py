"""Compare interpolation strengths (strong / mid / weak) across models 50x1-50x4.

**Selection**, per model: every (sample, pair) is scored in the full [0,1]^2
box exactly as in ``select_pairs.py`` (balance, size, boundary closeness,
soundness), but per *variant*, with the three ITP regions as the series and
their distinctness rewarded -- a good figure is one where the interpolants
differ visibly:

    score = sqrt(balance) * mean_variants[quality * (0.6 + 0.4 * distinct)] * ordinal penalty

A variant whose regions vanish in the slice contributes 0.

**Strong ITP does not contain its sample.**  The strong interpolants of
Direct and Guide-all (all samples) and of Fix-all (about half) miss their own
sample point by a little (median ~5e-6 on 50x1, up to ~2e-2 on 50x2), so a
slice through the sample misses them entirely.  Each explanation is therefore
sliced through *its nearest point to the sample* (L-inf LP; the sample itself
when it is contained).  Background and sample marker stay at the sample.  The
largest shift is stored as ``max_shift`` and printed in the figure title
whenever it is >= 1e-4.  Variants present
for >= 90% of samples are required (Direct is optional on 50x3 / 50x4, where
it mostly timed out).  The diverse top ``--top`` (default 20) per model are
picked, and **every pick is drawn on all four models**.

**Output** under ``plots/tacas/<dataset>/itp_compare/``:

  ``<net>/scores.csv``, ``<net>/selection.csv``  scoring of that model
  ``<net>/p<src>rNN_<dataset>-<net>_sSSS_xA-xB_<feat>_vs_<feat>_itp.{png,pdf}``
      1x3: Direct | Fix-all | Guide-all on model <net>, ITPs overlaid;
      ``p<src>rNN`` = picked on model <src> at rank NN
  ``combined/p<src>rNN_<dataset>_sSSS_xA-xB_..._itp_models.{png,pdf}``
      4x3: models (rows) x variants (columns)
  ``<net>/contact_sheet.png``  that model's 20 picks, Guide-all panel
  ``index.csv``  every pick and which models selected it

    PYTHONPATH=. python3 -m explviz.examples.tacas.itp_compare [--top 20] [--render-only]
"""

from __future__ import annotations

import argparse
import itertools
import math
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import linprog

from explviz import Panel, draw_panel
from explviz.geometry.halfplanes import close_ring
from explviz.geometry.region2d import regions_for_formula
from explviz.plot.legend import build_legend
from explviz.style import DEFAULT_STYLE

from .common import (DATASETS, ITPS, LABELS, axis_candidates, itp_dir, load_classifier,
                     load_points_for, load_polytopes, pair_stem)
from .render import save
from .select_pairs import (FULL, ORDINAL_LEVELS, ORDINAL_PENALTY, class_maps,
                           select_diverse, slice_metrics)

NETS = ["50x1", "50x2", "50x3", "50x4"]
VARIANTS = ["naive", "fix-all", "guide-all"]
#: drawn in this order, so the (usually larger) weak region sits underneath
ITP_ORDER = ["weak", "mid", "strong"]
ITP_LABELS = {"strong": "ITP strong", "mid": "ITP mid (factor 0.5)", "weak": "ITP weak"}
#: colour, marker, marker size, fill alpha, line width
ITP_SERIES = {"weak": ("tab:blue", "o", 12, 0.18, 4.0),
              "mid": ("tab:orange", "s", 9, 0.25, 2.5),
              "strong": ("tab:purple", "^", 8, 0.35, 1.5)}
REQUIRED_COVERAGE = 0.9
PIN_SLACK = 1e-3         # how far past the nearest point a pin may move inwards
SHIFT_NOTE = 5e-3        # note shifts above this in the title (visible scale)


def _deepest(p, x, rad):
    """Point within L-inf ``rad`` of ``x`` maximising the slack of every row.

    Returns (point, depth); depth 0 means the polytope has no interior there.
    """
    n = p.n_vars
    norms = np.linalg.norm(p.A_ub, axis=1)
    eye = np.eye(n)
    A = np.vstack([np.c_[p.A_ub, norms], np.c_[eye, np.zeros(n)], np.c_[-eye, np.zeros(n)]])
    b = np.r_[p.b_ub, x + rad, -(x - rad)]
    res = linprog(np.r_[np.zeros(n), -1.0], A_ub=A, b_ub=b,
                  bounds=[(0, 1)] * n + [(0, None)], method="highs")
    return (res.x[:n], float(res.x[-1])) if res.success else (None, -1.0)


def _nearest(p, x):
    """(point, L-inf distance) of the point of ``p`` closest to ``x``."""
    n = p.n_vars
    eye = np.eye(n)
    A = np.vstack([np.c_[p.A_ub, np.zeros(len(p.A_ub))],
                   np.c_[eye, -np.ones(n)], np.c_[-eye, -np.ones(n)]])
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=A, b_ub=np.r_[p.b_ub, x, -x],
                  bounds=[(0, 1)] * n + [(0, None)], method="highs")
    return (res.x[:n], float(res.x[-1])) if res.success else (None, np.inf)


def pin_point(polys, x):
    """(point, shift, degenerate): where to slice an explanation of sample ``x``.

    * contains ``x``: ``x`` itself.
    * misses ``x`` but has an interior: the deepest point within ``PIN_SLACK``
      of its nearest point, so the 2D slice has real area.
    * no interior at all (Direct / Guide-all strong ITP are single points):
      the nearest point, flagged ``degenerate`` -- drawn as a marker.
    """
    if any(p.contains(x, tol=1e-9) for p in polys):
        return x, 0.0, False
    best = (None, np.inf, True)
    for p in polys:
        near, dist = _nearest(p, x)
        if near is None:
            continue
        deep, depth = _deepest(p, x, dist + PIN_SLACK)
        if deep is not None and depth > 1e-9:
            cand = (deep, float(np.max(np.abs(deep - x))), False)
        else:
            cand = (near, dist, True)
        if (cand[2], cand[1]) < (best[2], best[1]):
            best = cand
    return best


def itp_style(ds, exps):
    keys = [e[0] for e in exps]
    col = lambda k: [ITP_SERIES[key][k] for key in keys] or [ITP_SERIES["weak"][k]]
    return DEFAULT_STYLE.derive(colors=col(0), markers=col(1), marker_sizes=col(2),
                                fill_alphas=col(3), line_widths=col(4),
                                class_labels=ds.class_labels)


def legend_labels(ds):
    return [*ds.class_labels, *(ITP_LABELS[k] for k in ITP_ORDER), DEFAULT_STYLE.point_label]


def load_net(ds, net, X):
    """(polys, pins): [variant][itp] -> per-sample list.

    ``polys`` is None where missing/unparsable; ``pins`` holds the
    (point, shift) each explanation is sliced through.
    """
    polys, pins = {}, {}
    for v in VARIANTS:
        polys[v], pins[v] = {}, {}
        for k in ITP_ORDER:
            P = load_polytopes(ds, net, v, ITPS[k])
            polys[v][k] = P
            pins[v][k] = [None if P[i] is None else pin_point(P[i], X[i])
                          for i in range(len(X))]
    return polys, pins


def present(polys, v, i):
    return all(polys[v][k][i] is not None for k in ITP_ORDER)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def score_net(ds, net, X, polys, pins):
    clf = load_classifier(ds, net)
    pred = clf.predict(X)
    axes = axis_candidates(X)
    pen = {j: 1.0 if len(np.unique(X[:, j])) >= ORDINAL_LEVELS else ORDINAL_PENALTY
           for j in axes}
    pairs = list(itertools.combinations(axes, 2))
    n = len(X)
    coverage = {v: np.mean([present(polys, v, i) for i in range(n)]) for v in VARIANTS}
    required = [v for v in VARIANTS if coverage[v] >= REQUIRED_COVERAGE]

    rows = []
    for i in range(n):
        if not all(present(polys, v, i) for v in required):
            continue
        variants = [v for v in VARIANTS if present(polys, v, i)]
        cmaps = class_maps(clf, X[i], pairs, [FULL] * len(pairs))
        for (a, b), cmap in zip(pairs, cmaps):
            row = {"sample": i, "a": a, "b": b, "methods": "+".join(variants),
                   "pred": int(pred[i])}
            per = []
            for v in variants:
                # single-point explanations have no area to score
                regions = {k: regions_for_formula(polys[v][k][i], (a, b), ds.spec.bounds,
                                                  "snapshot", point=pins[v][k][i][0])
                           for k in ITP_ORDER if not pins[v][k][i][2]}
                met = slice_metrics(cmap, pred[i], regions)
                if met is None:
                    per.append(0.0)
                    continue
                row["balance"] = met["balance"]
                row[f"{v}:quality"], row[f"{v}:distinct"] = met["quality"], met["distinct"]
                for k in regions:
                    row[f"{v}:{k}:area"] = met[f"{k}:area"]
                per.append(met["quality"] * (0.6 + 0.4 * met["distinct"]))
            row["max_shift"] = max(pins[v][k][i][1] for v in variants for k in ITP_ORDER)
            row["score"] = (np.sqrt(row.get("balance", 0.0)) * float(np.mean(per))
                            * pen[a] * pen[b])
            rows.append(row)
    return pd.DataFrame(rows), coverage


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def entries_for(net_data, i):
    """variant -> [(itp key, polys, pin point, shift, degenerate)] for sample i."""
    polys, pins = net_data
    return {v: [(k, polys[v][k][i], *pins[v][k][i]) for k in ITP_ORDER
                if polys[v][k][i] is not None and pins[v][k][i][0] is not None]
            for v in VARIANTS}


def shift_note(entries):
    shift = max((e[3] for exps in entries.values() for e in exps), default=0.0)
    return f" | strong-ITP pinned {shift:.0e} off sample" if shift >= SHIFT_NOTE else ""


def draw_itp_panel(ax, ds, x, a, b, exps, clf, title, style):
    """Background + sample via explviz, then each ITP sliced through its own pin."""
    draw_panel(ax, Panel(point=x, axes=(a, b), explanations=[], title=title),
               ds.spec, style=style, clf=clf)
    for j, (k, polys, pin, _, degenerate) in enumerate(exps):
        ser = style.series(j)
        if degenerate:
            ax.scatter([pin[a]], [pin[b]], marker="*", s=220, color=ser.color, zorder=101,
                       edgecolors="white", linewidths=0.6,
                       label=f"{ITP_LABELS[k]} (single point)")
            continue
        regions = regions_for_formula(polys, (a, b), ds.spec.bounds, "snapshot", point=pin)
        for n, r in enumerate(regions):
            ring = close_ring(r)
            ax.fill(ring[:, 0], ring[:, 1], color=ser.color, alpha=ser.alpha, zorder=1)
            ax.plot(ring[:, 0], ring[:, 1], color=ser.color, linewidth=ser.linewidth,
                    zorder=2, label=ITP_LABELS[k] if n == 0 else None)
            ax.scatter(r[:, 0], r[:, 1], marker=ser.marker, s=ser.size, color=ser.color,
                       zorder=2)
        if not regions:
            ax.plot([], [], color=ser.color, linewidth=ser.linewidth, label=ITP_LABELS[k])
    if not exps:
        ax.text(0.5, 0.5, "no explanation\n(timeout / unparsable)", ha="center",
                va="center", transform=ax.transAxes, fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))


def draw_grid(ds, nets, i, a, b, X, clfs, data, stem, suptitle, title=True, size=(5.2, 5.0)):
    """Rows = ``nets``, columns = variants, ITPs overlaid; one shared legend."""
    fig, axs = plt.subplots(len(nets), 3, squeeze=False,
                            figsize=(size[0] * 3 + 2.6, size[1] * len(nets)))
    note = ""
    for r, net in enumerate(nets):
        entries = entries_for(data[net], i)
        note = note or shift_note(entries)
        pred = int(clfs[net].predict(X[i])[0])
        for ax, v in zip(axs[r], VARIANTS):
            head = f"{net} (pred: {ds.class_labels[pred]}) | {LABELS[v]}" if len(nets) > 1 \
                else LABELS[v]
            draw_itp_panel(ax, ds, X[i], a, b, entries[v], clfs[net], head,
                           itp_style(ds, entries[v]).derive(label_fontsize=13, tick_fontsize=9,
                                                            point_size=30))
            ax.title.set_fontsize(12)
    build_legend(fig, list(axs.flat), order=legend_labels(ds), fontsize=12,
                 width_frac=0.14 if len(nets) > 1 else 0.17)
    fig.subplots_adjust(hspace=0.35)
    if title:
        fig.suptitle(suptitle + note, fontsize=12)
    out = save(fig, stem)
    plt.close(fig)
    return out


def contact_sheet(ds, net, sel, X, clf, net_data, ncols=5):
    n = len(sel)
    fig, axes = plt.subplots(math.ceil(n / ncols), ncols, squeeze=False,
                             figsize=(3.6 * ncols, 3.6 * math.ceil(n / ncols)))
    for ax, row in zip(axes.flat, sel.itertuples()):
        i, a, b = int(row.sample), int(row.a), int(row.b)
        exps = entries_for(net_data, i)["guide-all"]
        draw_itp_panel(ax, ds, X[i], a, b, exps, clf,
                       f"r{row.rank:02d} s{i:03d} x{a + 1}-x{b + 1} ({row.score:.2f})",
                       itp_style(ds, exps).derive(label_fontsize=8, tick_fontsize=6,
                                                  point_size=20))
        ax.title.set_fontsize(9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{ds.name} {net}: top {n} ITP picks, Guide-all panel "
                 f"(weak blue, mid orange, strong purple)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, itp_dir(ds) / net / "contact_sheet", formats=("png",), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--render-only", action="store_true",
                    help="reuse the saved selection.csv files")
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    for name in args.datasets:
        ds = DATASETS[name]
        X, _ = load_points_for(ds)
        data = {net: load_net(ds, net, X) for net in NETS}
        clfs = {net: load_classifier(ds, net) for net in NETS}

        sels = {}
        for net in NETS:
            out = itp_dir(ds) / net
            if args.render_only:
                sels[net] = pd.read_csv(out / "selection.csv")
                continue
            scores, coverage = score_net(ds, net, X, *data[net])
            sels[net] = select_diverse(scores, args.top)
            out.mkdir(parents=True, exist_ok=True)
            scores.sort_values("score", ascending=False).to_csv(out / "scores.csv", index=False)
            sels[net].to_csv(out / "selection.csv", index=False)
            print(name, net, "coverage", {v: round(c, 2) for v, c in coverage.items()},
                  "usable", int((scores.score > 0).sum()), "of", len(scores),
                  "| picks", sels[net]["score"].round(3).tolist())

        # one drawing per distinct (sample, pair); named after its first picker
        picks = {}
        for net in NETS:
            for r in sels[net].itertuples():
                key = (int(r.sample), int(r.a), int(r.b))
                picks.setdefault(key, {"src": net, "rank": int(r.rank), "score": r.score,
                                       "also": []})
                if picks[key]["src"] != net:
                    picks[key]["also"].append(f"{net}r{int(r.rank):02d}")

        index = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for (i, a, b), p in picks.items():
                tag = f"p{p['src']}r{p['rank']:02d}"
                note = f"picked on {p['src']}, rank {p['rank']}" + (
                    f" (also {', '.join(p['also'])})" if p["also"] else "")
                for net in NETS:
                    pred = int(clfs[net].predict(X[i])[0])
                    draw_grid(ds, [net], i, a, b, X, clfs, data,
                              itp_dir(ds) / net / f"{tag}_{ds.name}-{net}_"
                              f"{pair_stem(ds, i, a, b)[len(ds.name) + 1:]}_itp",
                              f"{ds.name} {net} | sample {i} (predicted: "
                              f"{ds.class_labels[pred]}) | {note}", title=not args.no_title)
                draw_grid(ds, NETS, i, a, b, X, clfs, data,
                          itp_dir(ds) / "combined" / f"{tag}_{pair_stem(ds, i, a, b)}_itp_models",
                          f"{ds.name} | sample {i} | {note} | full box",
                          title=not args.no_title, size=(4.6, 4.3))
                index.append(dict(tag=tag, sample=i, a=f"x{a + 1}", b=f"x{b + 1}",
                                  picked_on=p["src"], rank=p["rank"], score=round(p["score"], 3),
                                  also_picked_by=" ".join(p["also"])))
                print(name, tag, f"s{i:03d} x{a + 1}-x{b + 1}", note)
        for net in NETS:
            contact_sheet(ds, net, sels[net], X, clfs[net], data[net])
        pd.DataFrame(index).to_csv(itp_dir(ds) / "index.csv", index=False)


if __name__ == "__main__":
    main()
