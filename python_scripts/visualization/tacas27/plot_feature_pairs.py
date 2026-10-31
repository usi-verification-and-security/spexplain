"""One sample per dataset, its explanation in 4 feature planes (model 50x1, weak ITP).

Writes to ``plots/tacas27/``:

  ``feature_pair_selection.{png,pdf}``                     both rows, legend on top
  ``feature_pair_selection_<dataset>_sSSS.pdf``  one row each; only the first has the legend

Axes are labelled ``x<j>-<name>`` with real-world tick values; the planes are
still the scaled [0,1]^2 box, so nothing changes shape.

    PYTHONPATH=. python3 python_scripts/visualization/tacas27/plot_feature_pairs.py
"""

import warnings

import matplotlib.pyplot as plt
from explviz.plot.legend import collect_handles
from explviz.style import DEFAULT_STYLE

from common import (DATASETS, FINAL_FONT, ITP_STEMS, METHOD_LABELS, METHOD_SERIES, OUT_DIR,
                    axis_label, draw_methods_panel, real_ticks, save)

NET = "50x1"
#: one row each: (dataset, 0-based sample, method, the 4 planes as
#: (x-axis, y-axis) 1-based SMT variables)
PICKS = [("heart_attack", 20, "guide-all", [(1, 5), (5, 8), (1, 8), (10, 11)]),
         ("loan", 89, "guide-all", [(4, 7), (7, 8), (1, 8), (1, 5)])]
PANEL = 3.4             # inches per (square) panel
N_PAIRS = 4
#: every method gets Guide-all's line width (one explanation per panel)
LINE_WIDTH = DEFAULT_STYLE.line_widths[METHOD_SERIES["guide-all"]]


def draw(picks, stem, legend=True, formats=("png", "pdf")):
    """Rows = ``picks``; one legend row on top when ``legend``."""
    top, bottom, left = (0.8 if legend else 0.15), 0.6, 1.2       # margins, inches
    width = PANEL * N_PAIRS + 0.8
    # square axes come out ~2.55 in wide; the 1.25 in row gap holds the x labels
    height = 2.55 * len(picks) + 1.25 * (len(picks) - 1) + top + bottom
    fig, axes = plt.subplots(len(picks), N_PAIRS, squeeze=False, figsize=(width, height))
    for r, (name, i, method, pairs) in enumerate(picks):
        ds = DATASETS[name]
        X, clf = ds.points(), ds.classifier(NET)
        formula = {method: ds.formula(NET, method, ITP_STEMS["weak"], i)}
        for c, (a, b) in enumerate(pairs):
            a, b, ax = a - 1, b - 1, axes[r, c]
            draw_methods_panel(ax, ds, X[i], a, b, formula, clf, widths={method: LINE_WIDTH},
                               resolution=200, label_fontsize=FINAL_FONT["label"],
                               tick_fontsize=FINAL_FONT["tick"], point_size=60)
            ax.set_xlabel(axis_label(ds, a), fontsize=FINAL_FONT["label"], labelpad=2)
            ax.set_ylabel(axis_label(ds, b), fontsize=FINAL_FONT["label"], labelpad=2)
            ax.set_box_aspect(1)
            real_ticks(ax, ds, a, b, X)
        axes[r, 0].annotate(f"{ds.title}, s{i:03d}", xy=(-0.42, 0.5),
                            xycoords="axes fraction", ha="right", va="center", rotation=90,
                            fontsize=FINAL_FONT["row"], fontweight="bold")
    fig.subplots_adjust(left=left / width, right=0.995, bottom=bottom / height,
                        top=1 - top / height, wspace=0.42, hspace=1.25 / 2.55)

    if legend:
        # the datasets share class colours, so their class names are merged
        # ('High risk / Rejected'); over all PICKS, so a one-row file names them too
        names = list(dict.fromkeys(name for name, *_ in PICKS))
        merged = lambda c: " / ".join(dict.fromkeys(DATASETS[n].class_labels[c] for n in names))
        entries = [*((DATASETS[names[0]].class_labels[c], merged(c)) for c in (0, 1)),
                   *((METHOD_LABELS[m], METHOD_LABELS[m])
                     for m in dict.fromkeys(m for _, _, m, _ in PICKS)),
                   (DEFAULT_STYLE.point_label, DEFAULT_STYLE.point_label)]
        found = collect_handles(list(axes.flat))
        entries = [(found[key], text) for key, text in entries if key in found]
        fig.legend(*zip(*entries), loc="lower center", ncol=len(entries),
                   bbox_to_anchor=(0.5, axes[0, 0].get_position().y1 + 0.1 / height),
                   fontsize=FINAL_FONT["legend"], frameon=False, handlelength=1.5,
                   columnspacing=1.5, borderaxespad=0)
    written = save(fig, stem, formats=formats)
    plt.close(fig)
    return written


def main():
    print("wrote", *draw(PICKS, OUT_DIR / "feature_pair_selection"))
    for r, pick in enumerate(PICKS):
        name, i = pick[0], pick[1]
        print("wrote", *draw([pick], OUT_DIR / f"feature_pair_selection_{name}_s{i:03d}",
                             legend=r == 0, formats=("pdf",)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
