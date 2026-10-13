"""Draw every 50x1 pick on 50x4 and vice versa, to compare models on one sample.

Reads the selections written by ``select_pairs.py`` and, for every pick of a
source model, writes in the variant folder (``<dataset>/variant_compare``):

  ``<dst>/x<src>rNN_<dataset>-<dst>_sSSS_xA-xB_..._{overlay,grid}``
      the pick drawn on the other model, full [0,1]^2 box.  Skipped if the
      other model selected the same (sample, pair) itself.
  ``<src>/rNN_<dataset>-<src>_sSSS_xA-xB_..._full_{overlay,grid}``
      the pick redrawn on its own model in the full box, when the selected
      view was a zoom -- so that both models can be compared on one scale.
  ``cross_models/p<src>rNN_<dataset>_sSSS_xA-xB_..._models``
      both models (rows) x methods (columns) in one figure.

    PYTHONPATH=. python3 -m explviz.examples.tacas.cross_model
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from explviz import Explanation

from .common import (DATASETS, LABELS, METHODS, NETWORKS, figure_stem, load_classifier,
                     load_formulas, load_points_for, load_polytopes, pair_stem, variant_dir)
from .render import legend_order, model_grid, render_row, style_for

FULL = dict(view="full", win_a_lo=0.0, win_a_hi=1.0, win_b_lo=0.0, win_b_hi=1.0)


def drawable(ds, net):
    """method -> per-sample formula, ``None`` where missing or unparsable."""
    out = {}
    for m in METHODS:
        formulas, polys = load_formulas(ds, net, m), load_polytopes(ds, net, m)
        out[m] = [f if p is not None else None for f, p in zip(formulas, polys)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    for name in args.datasets:
        ds = DATASETS[name]
        X, _ = load_points_for(ds)
        clfs = {n: load_classifier(ds, n) for n in NETWORKS}
        formulas = {n: drawable(ds, n) for n in NETWORKS}
        sels = {n: pd.read_csv(variant_dir(ds) / n / "selection.csv") for n in NETWORKS}
        picked = {n: set(zip(s["sample"], s.a, s.b)) for n, s in sels.items()}
        log = []

        for src in NETWORKS:
            for dst in [n for n in NETWORKS if n != src]:
                for _, row in sels[src].iterrows():
                    i, a, b, rank = int(row["sample"]), int(row.a), int(row.b), int(row["rank"])
                    tag = f"x{src}r{rank:02d}"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # the pick on its own model, full box
                        if row.view != "full":
                            full = row.copy()
                            full.update(pd.Series(FULL))
                            render_row(ds, src, full, X, clfs[src], formulas[src],
                                       title=not args.no_title,
                                       stem=variant_dir(ds) / src /
                                       f"{figure_stem(ds, src, i, a, b, rank)}_full",
                                       note=f"full box (selected view: {row.view}) | "
                                            f"score {row.score:.2f}")
                        # the pick on the other model
                        methods = [m for m in METHODS if formulas[dst][m][i] is not None]
                        status = "skipped: also selected there" if (i, a, b) in picked[dst] \
                            else "skipped: no drawable explanation" if not methods else "drawn"
                        if status == "drawn":
                            other = row.copy()
                            other.update(pd.Series(FULL))
                            other["methods"] = "+".join(methods)
                            render_row(ds, dst, other, X, clfs[dst], formulas[dst],
                                       title=not args.no_title,
                                       stem=variant_dir(ds) / dst /
                                       f"{tag}_{figure_stem(ds, dst, i, a, b)}",
                                       note=f"full box | picked on {src}, rank {rank}")
                        # both models side by side
                        entries = {(n, m): [Explanation(formulas[n][m][i], LABELS[m], m)]
                                   for n in NETWORKS for m in METHODS
                                   if formulas[n][m][i] is not None}
                        model_grid(ds, NETWORKS, METHODS, i, a, b, X, clfs, entries,
                                   lambda exps: style_for(ds, [e.key for e in exps]),
                                   LABELS, legend_order(ds, METHODS),
                                   f"{ds.name} | sample {i} | picked on {src}, rank {rank} "
                                   f"| full box", variant_dir(ds) / "cross_models" /
                                   f"p{src}r{rank:02d}_{pair_stem(ds, i, a, b)}_models",
                                   title=not args.no_title)
                    log.append(dict(src=src, rank=rank, dst=dst, sample=i, a=a + 1, b=b + 1,
                                    selected_view=row.view, dst_methods="+".join(methods),
                                    status=status))
                    print(name, src, f"r{rank:02d}", "->", dst, status)
        pd.DataFrame(log).to_csv(variant_dir(ds) / "cross_models" / "index.csv", index=False)


if __name__ == "__main__":
    main()
