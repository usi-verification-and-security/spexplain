"""Weak ITP vs its unsat-core reductions (ucore, ucore_min), for Direct and Fix-all.

Writes ``plots/tacas27/heart_attack_ucore_weak_methods.{png,pdf}``: one row;
each pick under every setting in ``SETTINGS``, side by side.  ucore_min keeps
a subset of ucore's conjuncts, which keeps a subset of the ITP's, so the
regions are nested: ITP inside ucore inside ucore_min.  All three are sliced
through the ITP's pin (``common.pin_point``), so the overlay is exact.  The
panel title gives each one's #conjuncts.

    PYTHONPATH=. python3 python_scripts/visualization/tacas27/plot_ucore_methods.py
"""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from explviz import Panel, draw_panel
from explviz.plot.legend import build_legend
from explviz.style import DEFAULT_STYLE

from common import (DATASETS, FINAL_FONT, ITP_STEMS, LEGEND_W, METHOD_LABELS, OUT_DIR, PANEL_H,
                    draw_regions, panel_style, parse, pin_point, save)

DATASET = "heart_attack"
ITP = ITP_STEMS["weak"]
#: (net, method) settings, drawn next to each other for every pick
SETTINGS = [("50x1", "naive"), ("50x1", "fix-all")]
#: (0-based sample, x-axis, y-axis as 1-based SMT variables)
PICKS = [(83, 8, 11), (54, 5, 7)]

#: drawn in this order: the largest region underneath
KINDS = ["ucore_min", "ucore", "itp"]
KIND_LABELS = {"ucore_min": "ucore min", "ucore": "ucore", "itp": "ITP"}
#: colour, vertex marker, marker size, fill alpha, line width, line style
KIND_SERIES = {"ucore_min": ("#08519c", "o", 10, 0.15, 3.2, "--"),
               "ucore": ("#1b7837", "s", 9, 0.22, 2.4, "-"),
               "itp": ("#c51b7d", "^", 9, 0.35, 1.6, "-")}
#: order of the conjunct counts in panel titles
TITLE_KINDS = ["itp", "ucore", "ucore_min"]


def stem_of(kind):
    return ITP if kind == "itp" else f"{kind}_{ITP}"


def load_terms(ds, net, method, stem):
    """``#terms`` per sample from the run's stats file; ``None`` on timeout."""
    path = ds.explanation_file(net, method, stem).with_name(f"{stem}.stats.txt")
    out = []
    for line in path.read_text().splitlines():
        if line.startswith("sample ["):
            out.append(None)
        elif line.startswith("#terms:"):
            out[-1] = int(line.split()[1])
    return out


def undrawable_reason(formula):
    if re.search(r"\bl\d+_n\d+\b", formula):
        return "over hidden neurons"
    if "(ite" in formula:
        return "has ite terms"
    return "unparsable"


def draw_ucore_panel(ax, ds, net, method, X, i, a, b, clf):
    raw = {kind: ds.formula(net, method, stem_of(kind), i) for kind in KINDS}
    polys = {kind: parse(ds, raw[kind]) for kind in KINDS}
    terms = {kind: load_terms(ds, net, method, stem_of(kind))[i] for kind in KINDS}
    # formulas that exist but cannot be drawn (hidden neurons, ite terms)
    undrawable = {kind: undrawable_reason(raw[kind]) for kind in KINDS
                  if raw[kind] is not None and polys[kind] is None}

    # a point of the ITP is in all three, so they share its pin; when the ITP
    # is a single point, each is pinned on its own so the others keep their area
    pins = dict.fromkeys(KINDS)
    if polys["itp"] is not None:
        shared = pin_point(polys["itp"], X[i])
        for kind in KINDS:
            if polys[kind] is not None:
                pins[kind] = shared if not shared[2] else pin_point(polys[kind], X[i])

    counts = " · ".join(f"{kind} {terms[kind]}" if polys[kind] is not None or kind in undrawable
                        else f"{kind} timeout" for kind in TITLE_KINDS)
    title = f"sample {i} ({ds.predicted(clf, X[i])})\n{counts}"
    draw_panel(ax, Panel(point=X[i], axes=(a, b), explanations=[], title=title),
               ds.spec, style=panel_style(ds), clf=clf)
    ax.title.set_fontsize(FINAL_FONT["title"])

    for z, kind in enumerate(KINDS):
        if pins[kind] is None or pins[kind][0] is None:
            continue
        pin, _, degenerate = pins[kind]
        color, marker, size, alpha, width, style = KIND_SERIES[kind]
        if degenerate:
            # nested points coincide; different sizes so both show
            big = kind != "itp"
            ax.scatter([pin[a]], [pin[b]], marker="*", s=700 if big else 220,
                       color=color, zorder=101 + z, edgecolors="none" if big else "white",
                       linewidths=0.8, label=f"{KIND_LABELS[kind]} (single point)")
        else:
            draw_regions(ax, ds, polys[kind], a, b, pin, color, marker, size, alpha, width,
                         KIND_LABELS[kind], linestyle=style, z_fill=1 + z, z_line=5 + z)

    if undrawable:
        by_reason = {}
        for kind, reason in undrawable.items():
            by_reason.setdefault(reason, []).append(kind)
        ax.text(0.02, 0.98, "\n".join(f"{', '.join(kinds)}: {reason}, not drawn"
                                      for reason, kinds in by_reason.items()),
                transform=ax.transAxes, ha="left", va="top", fontsize=FINAL_FONT["note"],
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"), zorder=120)


def main():
    ds = DATASETS[DATASET]
    X = ds.points()
    clfs = {net: ds.classifier(net) for net, _ in SETTINGS}

    # explicit layout (inches): panels of one pick share the y-axis and sit
    # close together; picks are separated by room for tick labels + ylabel
    panel, gap_in, gap_out, left, right = 3.4, 0.3, 1, 0.5, LEGEND_W + 0.2
    ratios = []
    for p in range(len(PICKS)):
        for q in range(len(SETTINGS)):
            ratios.append(panel)
            if q < len(SETTINGS) - 1:
                ratios.append(gap_in)
        if p < len(PICKS) - 1:
            ratios.append(gap_out)
    width = left + sum(ratios) + right
    fig = plt.figure(figsize=(width, PANEL_H + 1.3))
    gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0, left=left / width,
                          right=1 - right / width, top=0.74, bottom=0.14)
    axes = np.array([[fig.add_subplot(gs[0, k]) for k, v in enumerate(ratios) if v == panel]])

    cells = [(pick, setting) for pick in PICKS for setting in SETTINGS]
    for c, (ax, ((i, a, b), (net, method))) in enumerate(zip(axes[0], cells)):
        draw_ucore_panel(ax, ds, net, method, X, i, a - 1, b - 1, clfs[net])
        # method header above the two-line title
        ax.annotate(METHOD_LABELS[method], xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 2.7 * FINAL_FONT["title"] + 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FINAL_FONT["row"], fontweight="bold")
        if c % len(SETTINGS):
            # same y-axis as the panel to its left
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    order = [*ds.class_labels,
             *(f"{KIND_LABELS[kind]}{s}" for kind in KINDS for s in ("", " (single point)")),
             DEFAULT_STYLE.point_label]
    build_legend(fig, list(axes.flat), order=order, fontsize=FINAL_FONT["legend"],
                 width_frac=LEGEND_W / width)
    print("wrote", *save(fig, OUT_DIR / f"{DATASET}_ucore_weak_methods"))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
