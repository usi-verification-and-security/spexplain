"""Fix-all, Guide-all, Direct and VeriX (eps=5%) on hand-picked slices of two datasets.

Writes ``plots/tacas27/variants_verix.{png,pdf}``: rows are the models,
columns the picks; a VeriX explanation that is a single point is a star.

    PYTHONPATH=. python3 python_scripts/visualization/tacas27/plot_variants_verix.py
"""

import warnings

import matplotlib.pyplot as plt
from explviz.plot.legend import collect_handles
from explviz.style import DEFAULT_STYLE

from common import (DATASETS, FINAL_FONT, ITP_STEMS, METHOD_LABELS, OUT_DIR, PANEL_FONT,
                    PANEL_H, PANEL_W, LEGEND_W, VERIX, draw_methods_panel, parse, save)

NETS = ["50x1", "50x4"]
#: (dataset, 0-based sample, x-axis, y-axis as 1-based SMT variables), in column order
PICKS = [("heart_attack", 97, 3, 4), ("heart_attack", 62, 4, 13),
         ("loan", 7, 5, 7), ("loan", 69, 1, 5)]
METHODS = ["fix-all", "guide-all", "naive", "verix"]
FIX_ALL_WIDTH = 3.0     # default fix-all width is 5


def formulas_of(ds, net, i):
    """method -> formula of sample ``i``, for the methods it can be drawn for."""
    out = {}
    for m in METHODS:
        f = ds.formula(net, m, VERIX if m == "verix" else ITP_STEMS["weak"], i)
        if parse(ds, f) is not None:
            out[m] = f
    return out


def main():
    names = list(dict.fromkeys(name for name, *_ in PICKS))
    X = {n: DATASETS[n].points() for n in names}

    fig, axes = plt.subplots(len(NETS), len(PICKS), squeeze=False,
                             figsize=(PANEL_W * len(PICKS) + LEGEND_W, PANEL_H * len(NETS)))
    for r, net in enumerate(NETS):
        clfs = {n: DATASETS[n].classifier(net) for n in names}
        for c, (name, i, a, b) in enumerate(PICKS):
            ds, x, ax = DATASETS[name], X[name][i], axes[r, c]
            draw_methods_panel(ax, ds, x, a - 1, b - 1, formulas_of(ds, net, i), clfs[name],
                               title=f"sample {i} ({ds.predicted(clfs[name], x)})",
                               widths={"fix-all": FIX_ALL_WIDTH}, **PANEL_FONT)
            ax.title.set_fontsize(FINAL_FONT["title"])
        # vertical, just left of the y-axis label
        axes[r, 0].annotate(net, xy=(-0.33, 0.5), xycoords="axes fraction", ha="right",
                            va="center", rotation=90, fontsize=FINAL_FONT["row"],
                            fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.08, top=0.9, hspace=0.38, wspace=0.3)

    # legend right next to the last column
    order = [*(lab for n in names for lab in DATASETS[n].class_labels),
             *(METHOD_LABELS[m] for m in METHODS), DEFAULT_STYLE.point_label]
    found = collect_handles(list(axes.flat))
    labels = [l for l in order if l in found]
    legend = fig.legend([found[l] for l in labels], labels, fontsize=FINAL_FONT["legend"],
                        frameon=True, loc="center left",
                        bbox_to_anchor=(axes[0, -1].get_position().x1 + 0.005, 0.5))
    for line, text in zip(legend.get_lines(), legend.get_texts()):
        if text.get_text() == METHOD_LABELS["fix-all"]:
            line.set_linewidth(FIX_ALL_WIDTH)

    # dataset name above its columns
    for name in names:
        cols = [c for c, (n, *_) in enumerate(PICKS) if n == name]
        pos0, pos1 = axes[0, cols[0]].get_position(), axes[0, cols[-1]].get_position()
        fig.text((pos0.x0 + pos1.x1) / 2, pos0.y1 + 0.055, DATASETS[name].title, ha="center",
                 va="bottom", fontsize=FINAL_FONT["row"], fontweight="bold")
    print("wrote", *save(fig, OUT_DIR / "variants_verix"))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
