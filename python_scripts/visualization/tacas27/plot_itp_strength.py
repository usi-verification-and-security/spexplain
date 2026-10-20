"""ITP strength (Guide-all): weak / mid / strong ITP overlaid, models in rows.

Writes ``plots/tacas27/heart_attack_itp.{png,pdf}``.  Each ITP is sliced
through its own pin (``common.pin_point``): strong ITPs often miss their sample
by a little and are then a single point, drawn as a star.

    PYTHONPATH=. python3 python_scripts/visualization/tacas27/plot_itp_strength.py
"""

import warnings

import matplotlib.pyplot as plt
from explviz import Panel, draw_panel
from explviz.plot.legend import build_legend
from explviz.style import DEFAULT_STYLE

from common import (DATASETS, FINAL_FONT, ITP_STEMS, LEGEND_W, OUT_DIR, PANEL_H, PANEL_W,
                    draw_regions, panel_style, parse, pin_point, save)

DATASET = "heart_attack"
METHOD = "guide-all"
NETS = ["50x1", "50x4"]
#: (rank in the selection, 0-based sample, x-axis, y-axis as 1-based SMT variables)
PICKS = [(1, 83, 8, 11), (2, 62, 4, 13), (14, 40, 7, 10), (15, 9, 8, 11)]

#: drawn in this order, so the (usually larger) weak region sits underneath
ITP_ORDER = ["weak", "mid", "strong"]
ITP_LABELS = {"weak": "ITP weak", "mid": "ITP mid (factor 0.5)", "strong": "ITP strong"}
#: colour, marker, marker size, fill alpha, line width
ITP_SERIES = {"weak": ("tab:blue", "o", 12, 0.18, 4.0),
              "mid": ("tab:orange", "s", 9, 0.25, 2.5),
              "strong": ("tab:purple", "^", 8, 0.35, 1.5)}


def draw_itp_panel(ax, ds, net, X, i, a, b, clf, title):
    draw_panel(ax, Panel(point=X[i], axes=(a, b), explanations=[], title=title),
               ds.spec, style=panel_style(ds), clf=clf)
    ax.title.set_fontsize(FINAL_FONT["title"])
    drawn = False
    for k in ITP_ORDER:
        polys = parse(ds, ds.formula(net, METHOD, ITP_STEMS[k], i))
        if polys is None:
            continue
        pin, _, degenerate = pin_point(polys, X[i])
        if pin is None:
            continue
        drawn = True
        color, marker, size, alpha, width = ITP_SERIES[k]
        if degenerate:
            ax.scatter([pin[a]], [pin[b]], marker="*", s=220, color=color, zorder=101,
                       edgecolors="white", linewidths=0.6,
                       label=f"{ITP_LABELS[k]} (single point)")
        else:
            draw_regions(ax, ds, polys, a, b, pin, color, marker, size, alpha, width,
                         ITP_LABELS[k])
    if not drawn:
        ax.text(0.5, 0.5, "no explanation\n(timeout / unparsable)", ha="center",
                va="center", transform=ax.transAxes, fontsize=11,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))


def main():
    ds = DATASETS[DATASET]
    X = ds.points()
    width = PANEL_W * len(PICKS) + LEGEND_W
    fig, axes = plt.subplots(len(NETS), len(PICKS), squeeze=False,
                             figsize=(width, PANEL_H * len(NETS)))
    for r, net in enumerate(NETS):
        clf = ds.classifier(net)
        for c, (rank, i, a, b) in enumerate(PICKS):
            title = f"sample {i} ({ds.predicted(clf, X[i])})"
            if r == 0:
                title = f"r{rank:02d} | {title}"
            draw_itp_panel(axes[r, c], ds, net, X, i, a - 1, b - 1, clf, title)
        axes[r, 0].annotate(net, xy=(-0.38, 0.5), xycoords="axes fraction", ha="right",
                            va="center", fontsize=FINAL_FONT["row"], fontweight="bold")

    order = [*ds.class_labels, *(ITP_LABELS[k] for k in ITP_ORDER),
             f"{ITP_LABELS['strong']} (single point)", DEFAULT_STYLE.point_label]
    build_legend(fig, list(axes.flat), order=order, fontsize=FINAL_FONT["legend"],
                 width_frac=LEGEND_W / width)
    fig.subplots_adjust(left=0.07, hspace=0.35, wspace=0.35, top=0.86)
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    fig.text((x0 + x1) / 2, 0.955, "ITP strength (Guide-all)", ha="center", va="center",
             fontsize=FINAL_FONT["row"], fontweight="bold")
    print("wrote", *save(fig, OUT_DIR / f"{DATASET}_itp"))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
