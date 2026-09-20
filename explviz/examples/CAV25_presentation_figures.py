"""Reproduces the three talk figures in plotings/presentation/.

Each was its own ~290-line script; here they are three configurations of the
same `plot_grid` call.  Set ``USETEX = True`` to get the LaTeX labels the
originals used (requires a working LaTeX install).

    PYTHONPATH=. python3 -m explviz.examples.CAV25_presentation_figures
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from explviz import (Classifier, Explanation, Panel, load_explanations,
                     load_points, plot_grid, save_figure)
from explviz.paths import data_path
from explviz.specs import HEART_ATTACK_RAW as SPEC
from explviz.style import DEFAULT_STYLE

USETEX = False
DATASET = "heartAttack.csv"
MODEL = "heart_attack-50.nnet"
#: NB: paths below name the repo-root ``examples/`` data directory,
#: not this ``explviz/examples/`` package directory.

TEX = {
    "weak": r"\texttt{weak}",
    "mid": r"\texttt{mid}",
    "capture": r"$\textrm{\textbf{Capture}}$",
    "rmin": r"$\ensuremath{\textrm{\textbf{R}}_\mathit{min}} \circ \textrm{\textbf{Capture}}$",
    "generalize": r"$\textrm{\textbf{Generalize}}$",
    "interval": "Interval",
}
PLAIN = {"weak": "weak", "mid": "mid", "capture": "Capture",
         "rmin": "R_min o Capture", "generalize": "Generalize",
         "interval": "Interval"}


def L(key):
    return (TEX if USETEX else PLAIN)[key]


# Each figure: (name, panel list builder)
# panels are (sample index, 1-based axis pair)
ITPS_PANELS = [(107, (1, 4)), (196, (1, 4)), (107, (5, 8))]
CAPTURE_PANELS = [(107, (1, 4)), (196, (1, 4)), (107, (5, 8))]
GENERALIZE_PANELS = [(39, (3, 12)), (196, (1, 4)), (107, (5, 8))]


def fixed_source_panels(X, sources, panels, mode, background="auto"):
    """Panels where every method has one file for all axis pairs."""
    loaded = [(key, load_explanations(data_path(path))) for key, path in sources]
    out = []
    for exp_no, (a, b) in panels:
        out.append(Panel(
            point=X[exp_no], axes=(a - 1, b - 1), mode=mode, background=background,
            explanations=[Explanation(F[exp_no], L(key), key) for key, F in loaded],
        ))
    return out


def per_axis_source_panels(X, prefixes, panels, mode, background="auto"):
    """Panels where the file name depends on the axis pair (``..._vars_xA_xB``)."""
    out = []
    for exp_no, (a, b) in panels:
        exps = []
        for key, prefix in prefixes:
            F = load_explanations(data_path(f"{prefix}x{a}_x{b}.phi.txt"))
            exps.append(Explanation(F[exp_no], L(key), key))
        out.append(Panel(point=X[exp_no], axes=(a - 1, b - 1), mode=mode,
                         background=background, explanations=exps))
    return out


def main():
    X, _ = load_points(data_path(DATASET))
    clf = Classifier.from_nnet(data_path(MODEL))
    style = DEFAULT_STYLE.derive(usetex=USETEX)

    for background in ("contour", "scatter"):
        # --- itps: interpolation strategies, full 13-var formulas -> snapshot
        fig = plot_grid(fixed_source_panels(
            X,
            [("weak", "examples/ucores/updated_data_for_itp_strategies/itp_aweak_bstrong.phi.txt"),
             ("mid", "examples/ucores/updated_data_for_itp_strategies/itp_afactor_0.5_bstrong.phi.txt")],
            ITPS_PANELS, mode="snapshot", background=background,
        ), SPEC, ncols=3, style=style, clf=clf, seed=0)
        print("wrote", *save_figure(fig, f"plots/explviz/presentation/itps_{background}"))

        # --- generalize: generalized explanation vs plain interval
        fig = plot_grid(fixed_source_panels(
            X,
            [("generalize", "examples/full_heart_attack/itp__trial_n_4_abductive.phi.txt"),
             ("interval", "examples/full_heart_attack/trial_n_4_abductive.phi.txt")],
            GENERALIZE_PANELS, mode="snapshot", background=background,
        ), SPEC, ncols=3, style=style, clf=clf, seed=0)
        print("wrote", *save_figure(fig, f"plots/explviz/presentation/generalize_{background}"))

    # --- capture: per-axis-pair files, drawn as projections.
    # Projection mode shows no class regions, so there is only one version of
    # this figure and it needs no classifier.
    fig = plot_grid(per_axis_source_panels(
        X,
        [("rmin", "examples/partial/ucore_min_itp_vars_"),
         ("capture", "examples/partial/itp_vars_")],
        CAPTURE_PANELS, mode="projection",
    ), SPEC, ncols=3, style=style)
    print("wrote", *save_figure(fig, "plots/explviz/presentation/capture"))


if __name__ == "__main__":
    main()
