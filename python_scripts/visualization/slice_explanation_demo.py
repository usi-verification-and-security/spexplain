"""Demo of ``slice_formula``: cut one explanation down to a 2D SMT2 formula.

Takes sample 71's heart-attack explanation, pins every feature except Oldpeak
(x10) and Age (x1) to the sample's values, prints the sliced formula, saves it
as a one-line ``.phi.txt``, and plots it side by side with the original.  The
two panels should show the same region: the original is drawn as a snapshot
(which slices it on the fly), the sliced formula already has only two variables.

Run from the repo root:

    PYTHONPATH=. python3 python_scripts/visualization/slice_explanation_demo.py
"""

from pathlib import Path

import numpy as np

from explviz import (Classifier, Explanation, Panel, load_explanations,
                     load_points, parse_formula, plot_grid, region_2d,
                     save_figure, slice_formula)
from explviz.paths import data_path
from explviz.specs import HEART_ATTACK_SCALED as spec

SAMPLE = 71
AXES = (9, 0)                  # 0-based: Oldpeak (x10) on x, Age (x1) on y
EXP = "explanations/heart_attack/50x1/s100_scaled/fix-all/itp_aweak_bstrong.phi.txt"
MODEL = "models/heart_attack/heart_attack_50x1.nnet"
DATA = "datasets/heart_attack/heart_attack_s100_scaled.csv"
OUT = Path(__file__).parent / "plots" / f"ha_s{SAMPLE}_sliced_x10_x1"


def main():
    X, _ = load_points(data_path(DATA))
    original = load_explanations(data_path(EXP))[SAMPLE]
    point = X[SAMPLE]

    sliced = slice_formula(original, spec.var_names, point, AXES, fixed_vars=True)
    print(sliced)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    phi = OUT.with_suffix(".phi.txt")
    phi.write_text(sliced + "\n")
    print(f"\n{len(original)} -> {len(sliced)} characters; wrote {phi}")

    # Numeric check: both formulas give the same snapshot polygon.
    regions = [region_2d(p, AXES, spec.bounds, "snapshot", point=point)
               for f in (original, sliced) for p in parse_formula(f, spec.var_names)]
    print("same region:", all(r is not None for r in regions)
          and np.allclose(regions[0], regions[1]))

    panels = [
        Panel(point=point, axes=AXES, mode="snapshot", title=title,
              explanations=[Explanation(formula, label, "fix-all")])
        for formula, label, title in [
            (original, "Fix-all", "Original (13 variables)"),
            (sliced, "Fix-all", "Sliced (x1, x10 only)"),
        ]
    ]
    clf = Classifier.from_nnet(data_path(MODEL))
    fig = plot_grid(panels, spec, ncols=2, clf=clf)
    for written in save_figure(fig, OUT):
        print(written)


if __name__ == "__main__":
    main()
