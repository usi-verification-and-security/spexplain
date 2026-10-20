"""Reproduces plotings/ATVA_paper/methods-compare.py.

A 1x3 grid comparing five explanation methods on the 50x1 heart-attack model.
Compare this file's length with the 332-line original: everything here is
configuration, and the drawing lives in the package.

    PYTHONPATH=. python3 -m explviz.examples.ATVA26_methods_compare
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from explviz import (Classifier, Panel, explanations_from_formulas, load_points,
                     load_explanations, plot_grid, save_figure)
from explviz.datasets import select_explanation
from explviz.paths import data_path, explanations_path
from explviz.specs import HEART_ATTACK_SCALED as SPEC
from explviz.style import DEFAULT_STYLE

MODEL = "50x1"
VARIANT = "itp_aweak_bstrong"
METHODS = ["fix-all", "fix-guided", "prefer-all", "naive", "verix"]

#: VeriX only produced explanations for a few samples, in its own order.
INDEX_MAPS = {"verix": {0: 0, 1: 1, 71: 2, 84: 3, 6: 4}}

LABELS = {
    "naive": "Direct",
    "fix-all": "Fix-all",
    "prefer-all": "Guide-all",
    "fix-guided": "Hybrid",
    "verix": r"VeriX - CS - $\epsilon=5\%$",
}

#: (sample index, 1-based axis pair) for each panel.
PANEL_SPECS = [(71, (10, 1)), (71, (8, 10)), (6, (4, 1))]

LEGEND_ORDER = ["High risk", "Low risk", "Direct", "Guide-all", "Fix-all",
                "Hybrid", LABELS["verix"], "Sample point"]


def build_panels(background="contour", mode="snapshot"):
    X, _ = load_points(data_path("datasets/heart_attack_s100_scaled.csv"))
    sources = {m: load_explanations(
        explanations_path(f"HA/{MODEL}/{m}/{VARIANT}.phi.txt"))
        for m in METHODS}

    panels = []
    for exp_no, (a, b) in PANEL_SPECS:
        formulas = [select_explanation(sources[m], exp_no, INDEX_MAPS.get(m))
                    for m in METHODS]
        panels.append(Panel(
            point=X[exp_no],
            axes=(a - 1, b - 1),
            explanations=explanations_from_formulas(formulas, METHODS, LABELS),
            mode=mode,
            background=background,
        ))
    return panels


def main():
    clf = Classifier.from_nnet(data_path(f"models/heart_attack_{MODEL}.nnet"))
    style = DEFAULT_STYLE.derive(usetex=False)

    for background in ("contour", "scatter"):
        fig = plot_grid(build_panels(background), SPEC, ncols=3, style=style,
                        clf=clf, legend_order=LEGEND_ORDER,
                        repair=True, seed=0)
        out = save_figure(fig, f"plots/explviz/{MODEL}/methods_compare_{background}")
        print("wrote", *out)


if __name__ == "__main__":
    main()
