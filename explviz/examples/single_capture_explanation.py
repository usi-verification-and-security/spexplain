"""Reproduces plotings/demo_paper/single_db.py: one explanation, one axis pair.

The explanation files here are named ``..._vars_x<a>_x<b>.phi.txt``.  They are
still 13-variable formulas computed *for* that axis pair, so the original drew
them without pinning anything -- i.e. as a projection.

    PYTHONPATH=. python3 -m explviz.examples.single_capture_explanation
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from explviz import (Explanation, load_explanations, load_points,
                     plot_explanation, save_figure)
from explviz.paths import data_path
from explviz.specs import HEART_ATTACK_RAW
from explviz.style import DEFAULT_STYLE

SAMPLE = 107
AXES = (5, 8)          # 1-based, as in the original
#: NB: the repo-root ``examples/`` data directory, not ``explviz/examples/``.
PREFIX = "examples/partial/ucore_min_itp_vars_"

#: single_db.py capped cholesterol at 400 rather than the full 564.
SPEC = HEART_ATTACK_RAW.__class__(
    name="heart-attack-raw-chol400",
    var_names=HEART_ATTACK_RAW.var_names,
    axis_labels=HEART_ATTACK_RAW.axis_labels,
    bounds=[(lo, 400.0) if i == 4 else (lo, hi)
            for i, (lo, hi) in enumerate(HEART_ATTACK_RAW.bounds)],
    scaled=False,
)


def main():
    a, b = AXES
    X, _ = load_points(data_path("heartAttack.csv"))
    formulas = load_explanations(data_path(f"{PREFIX}x{a}_x{b}.phi.txt"))

    exp = Explanation(formula=formulas[SAMPLE], label="Explanation", key="explanation")
    style = DEFAULT_STYLE.derive(colors=["blue"], line_widths=[2], fill_alphas=[0.25],
                                 marker_sizes=[10])

    # Projection mode draws no class regions, so no classifier is needed here:
    # the shadow of the polytope is not comparable to a slice through the point.
    fig, ax = plot_explanation(
        exp, X[SAMPLE], (a - 1, b - 1), SPEC,
        mode="projection", style=style, figsize=(8, 6),
    )
    ax.legend(fontsize=style.legend_fontsize)
    fig.tight_layout()
    out = save_figure(fig, f"plots/explviz/single/exp{SAMPLE}_x{a}_x{b}")
    print("wrote", *out)


if __name__ == "__main__":
    main()
