"""Draw the slices chosen by ``select_pairs.py``.

For every row of ``plots/tacas/<dataset>/<net>/selection.csv``:

  ``<stem>_overlay.{png,pdf}``  all methods on one snapshot
  ``<stem>_grid.{png,pdf}``     one panel per method, same window

plus ``contact_sheet.png`` per network (all overlays as thumbnails) for a quick
manual pass.  ``<stem>`` is :func:`common.figure_stem`, e.g.
``r01_heart_attack-50x1_s073_x1-x10_age_vs_oldpeak``.

The plot window is the one the selection scored (``view`` column): the full
[0,1]^2 box, or a zoom around the explanations and the nearest boundary.

    PYTHONPATH=. python3 -m explviz.examples.tacas.render [--nets 50x1] [--ranks 1 2 3]
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from explviz import Explanation, Panel, draw_panel
from explviz.plot.legend import build_legend
from explviz.style import DEFAULT_STYLE

from .common import (DATASETS, LABELS, METHOD_SERIES, METHODS, NETWORKS, OUT_DIR, variant_dir,
                     figure_stem, load_classifier, load_formulas, load_points_for)


def save(fig, stem, formats=("png", "pdf"), dpi=200):
    """Like explviz.save_figure, but the tight box also keeps the suptitle."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = [stem.with_suffix(f".{f}") for f in formats]
    for path in out:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return out


def style_for(ds, methods):
    """A Style whose series i is the fixed appearance of ``methods[i]``."""
    base = DEFAULT_STYLE
    idx = [METHOD_SERIES[m] for m in methods]
    pick = lambda seq: [seq[i % len(seq)] for i in idx]
    return base.derive(colors=pick(base.colors), markers=pick(base.markers),
                       marker_sizes=pick(base.marker_sizes), fill_alphas=pick(base.fill_alphas),
                       line_widths=pick(base.line_widths), class_labels=ds.class_labels,
                       label_fontsize=15, legend_fontsize=14)


def windowed_spec(ds, row):
    bounds = list(ds.spec.bounds)
    bounds[int(row.a)] = (row.win_a_lo, row.win_a_hi)
    bounds[int(row.b)] = (row.win_b_lo, row.win_b_hi)
    return replace(ds.spec, bounds=bounds)


def legend_order(ds, methods):
    return [*ds.class_labels, *(LABELS[m] for m in methods), DEFAULT_STYLE.point_label]


def header(ds, net, row, pred, note=None):
    tail = note or f"view: {row.view} | score {row.score:.2f}"
    return (f"{ds.name} {net} | sample {int(row['sample'])} "
            f"(predicted: {ds.class_labels[pred]}) | {tail}")


def render_row(ds, net, row, X, clf, formulas, title=True, stem=None, note=None):
    """Overlay + grid figures of one selection row; ``stem``/``note`` override
    the default file name and the header's view/score part."""
    i, a, b = int(row["sample"]), int(row.a), int(row.b)
    methods = [m for m in METHODS if m in row.methods.split("+")]
    spec = windowed_spec(ds, row)
    exps = [Explanation(formulas[m][i], LABELS[m], m) for m in methods]
    pred = int(clf.predict(X[i])[0])
    stem = stem or variant_dir(ds) / net / figure_stem(ds, net, i, a, b, int(row["rank"]))
    written = []

    # overlay: every method on one Axes
    style = style_for(ds, methods)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    draw_panel(ax, Panel(point=X[i], axes=(a, b), explanations=exps), spec,
               style=style, clf=clf)
    build_legend(fig, [ax], order=legend_order(ds, methods),
                 fontsize=style.legend_fontsize, width_frac=0.28)
    if title:
        fig.suptitle(header(ds, net, row, pred, note), fontsize=11)
    written += save(fig, f"{stem}_overlay")
    plt.close(fig)

    # grid: one method per panel, same window and background
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods) + 2.5, 5),
                             squeeze=False)
    for ax, m, e in zip(axes[0], methods, exps):
        draw_panel(ax, Panel(point=X[i], axes=(a, b), explanations=[e], title=LABELS[m]),
                   spec, style=style_for(ds, [m]), clf=clf)
    build_legend(fig, list(axes[0]), order=legend_order(ds, methods),
                 fontsize=style.legend_fontsize, width_frac=0.17)
    if title:
        fig.suptitle(header(ds, net, row, pred, note), fontsize=11)
    written += save(fig, f"{stem}_grid")
    plt.close(fig)
    return written


def model_grid(ds, nets, cols, i, a, b, X, clfs, entries, style_fn, col_labels,
               legend_labels, suptitle, stem, title=True):
    """Rows = models, columns = ``cols``; full [0,1]^2 box everywhere.

    ``entries[(net, col)]`` is the list of Explanations drawn in that panel and
    ``style_fn(exps)`` its Style.  Each panel title states the model's own
    prediction, since a sample can be classified differently by each model.
    """
    fig, axes = plt.subplots(len(nets), len(cols), squeeze=False,
                             figsize=(4.6 * len(cols) + 2.6, 4.3 * len(nets)))
    for r, net in enumerate(nets):
        pred = int(clfs[net].predict(X[i])[0])
        for c, col in enumerate(cols):
            ax = axes[r, c]
            exps = entries.get((net, col), [])
            style = style_fn(exps).derive(label_fontsize=12, tick_fontsize=9, point_size=30)
            draw_panel(ax, Panel(point=X[i], axes=(a, b), explanations=exps,
                                 title=f"{net} (pred: {ds.class_labels[pred]}) | {col_labels[col]}"),
                       ds.spec, style=style, clf=clfs[net])
            ax.title.set_fontsize(11)
            if not exps:
                ax.text(0.5, 0.5, "no explanation\n(timeout / unparsable)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=11,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))
    build_legend(fig, list(axes.flat), order=legend_labels, fontsize=12, width_frac=0.14)
    fig.subplots_adjust(hspace=0.35)
    if title:
        fig.suptitle(suptitle, fontsize=13)
    out = save(fig, stem)
    plt.close(fig)
    return out


def contact_sheet(ds, net, sel, X, clf, formulas, ncols=5):
    n = len(sel)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax, row in zip(axes.flat, sel.itertuples()):
        i, a, b = int(row.sample), int(row.a), int(row.b)
        methods = [m for m in METHODS if m in row.methods.split("+")]
        style = style_for(ds, methods).derive(label_fontsize=8, tick_fontsize=6,
                                              point_size=20)
        exps = [Explanation(formulas[m][i], LABELS[m], m) for m in methods]
        draw_panel(ax, Panel(point=X[i], axes=(a, b), explanations=exps,
                             title=f"r{row.rank:02d} s{i:03d} x{a + 1}-x{b + 1} "
                                   f"{row.view} ({row.score:.2f})"),
                   windowed_spec(ds, row), style=style, clf=clf, resolution=120)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{ds.name} {net}: top {n} slices (overlay; colours: Fix-all red, "
                 f"Guide-all green, Direct blue)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = save(fig, variant_dir(ds) / net / "contact_sheet", formats=("png",), dpi=110)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--nets", nargs="+", default=NETWORKS)
    ap.add_argument("--ranks", nargs="+", type=int, help="only these selection ranks")
    ap.add_argument("--no-title", action="store_true", help="paper-ready: no header line")
    args = ap.parse_args()

    for name in args.datasets:
        ds = DATASETS[name]
        X, _ = load_points_for(ds)
        for net in args.nets:
            sel = pd.read_csv(variant_dir(DATASETS[name]) / net / "selection.csv")
            if args.ranks:
                sel = sel[sel["rank"].isin(args.ranks)]
            clf = load_classifier(ds, net)
            formulas = {m: load_formulas(ds, net, m) for m in METHODS}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for row in sel.itertuples(index=False):
                    for path in render_row(ds, net, pd.Series(row._asdict()), X, clf,
                                           formulas, title=not args.no_title):
                        if path.suffix == ".png":
                            print("wrote", path.relative_to(OUT_DIR))
                if not args.ranks:
                    print("wrote", *contact_sheet(ds, net, sel, X, clf, formulas))
            for w in {str(w.message) for w in caught if "contain" in str(w.message)}:
                print("WARNING:", w[:200])


if __name__ == "__main__":
    main()
