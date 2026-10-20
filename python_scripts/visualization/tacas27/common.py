"""Shared code for the final figures drawn by the plot_*.py scripts in this folder.

Datasets and where their files live, loading one explanation of one sample,
the per-method appearance, slicing an explanation that misses its sample
(``pin_point``), and the small drawing helpers the figures have in common.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy.optimize import linprog

from explviz import (Classifier, DatasetSpec, Explanation, Panel, draw_panel,
                     load_explanations, load_points, parse_formula)
from explviz.geometry.halfplanes import close_ring
from explviz.geometry.region2d import regions_for_formula
from explviz.parsing.smt2 import FormulaError
from explviz.paths import data_path
from explviz.specs import HEART_ATTACK_LABELS, HEART_ATTACK_LOWER, HEART_ATTACK_UPPER
from explviz.style import DEFAULT_STYLE

OUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "tacas27"

#: explanation file stems
ITP_STEMS = {"weak": "itp_aweak_bstrong", "mid": "itp_afactor_0.5_bstrong",
             "strong": "itp_astrong_bstrong"}
VERIX = "verix_eps_5perc"

#: font sizes shared by all final figures
FINAL_FONT = dict(label=16, tick=14, title=15, legend=16, row=18, note=13)
PANEL_FONT = dict(label_fontsize=FINAL_FONT["label"], tick_fontsize=FINAL_FONT["tick"],
                  point_size=40)
#: panel size (inches) and legend strip width
PANEL_W, PANEL_H, LEGEND_W = 4.3, 4.3, 3.2

METHOD_LABELS = {"naive": "Direct", "fix-all": "Fix-all", "guide-all": "Guide-all",
                 "verix": "VeriX (ε=5%)"}
#: index into DEFAULT_STYLE's series lists: fix-all red, guide-all green,
#: direct blue, verix purple -- the same in every figure
METHOD_SERIES = {"fix-all": 0, "guide-all": 2, "naive": 3, "verix": 4}

LOAN_LABELS = [
    "Age", "Income", "Employment exp.", "Loan amount", "Interest rate",
    "Loan / income", "Credit history length", "Credit score", "Gender: male",
    "Edu: Bachelor", "Edu: Doctorate", "Edu: High school", "Edu: Master",
    "Home: other", "Home: own", "Home: rent", "Intent: education",
    "Intent: home impr.", "Intent: medical", "Intent: personal",
    "Intent: venture", "Previous default",
]
#: real-world (unscaled) range per loan feature; the 14 one-hot features are 0/1
LOAN_BOUNDS = [(20.0, 94.0), (8000.0, 2448661.0), (0.0, 76.0), (500.0, 35000.0),
               (5.42, 20.0), (0.0, 0.66), (2.0, 30.0), (390.0, 784.0)] + [(0.0, 1.0)] * 14


def _scaled_spec(name, labels):
    n = len(labels)
    return DatasetSpec(name=name, var_names=[f"x{i}" for i in range(1, n + 1)],
                       axis_labels=list(labels), bounds=[(0.0, 1.0)] * n, scaled=True)


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    spec: DatasetSpec
    class_labels: tuple  # (class 0, class 1)
    real_bounds: list    # real-world range per feature, for the tick labels

    def points(self):
        return load_points(data_path(f"datasets/{self.name}/{self.name}_s100_scaled.csv"))[0]

    def classifier(self, net):
        return Classifier.from_nnet(data_path(f"models/{self.name}/{self.name}_{net}.nnet"))

    def predicted(self, clf, x):
        return self.class_labels[int(clf.predict(x)[0])]

    def explanation_file(self, net, method, stem):
        return data_path(f"explanations/{self.name}/{net}/s100_scaled/{method}/{stem}.phi.txt",
                         must_exist=False)

    def formula(self, net, method, stem, i):
        """Formula of sample ``i``; ``None`` if the file is missing or the run
        produced no formula (``<null>``)."""
        path = self.explanation_file(net, method, stem)
        if not path.exists():
            return None
        f = load_explanations(path)[i]
        return f if f.lstrip().startswith("(") else None


DATASETS = {
    "heart_attack": Dataset("heart_attack", "Heart attack",
                            _scaled_spec("heart-attack-scaled", HEART_ATTACK_LABELS),
                            tuple(DEFAULT_STYLE.class_labels),
                            list(zip(HEART_ATTACK_LOWER, HEART_ATTACK_UPPER))),
    "loan": Dataset("loan", "Loan", _scaled_spec("loan-scaled", LOAN_LABELS),
                    ("Rejected", "Approved"), LOAN_BOUNDS),
}


_NEURON_VAR = re.compile(r"\bl\d+_n\d+\b")


def parse(ds, formula):
    """Parsed polytopes, or ``None`` for a missing formula or one explviz cannot
    draw (``ite`` terms, or variables over hidden neurons)."""
    if formula is None or _NEURON_VAR.search(formula):
        return None
    try:
        return parse_formula(formula, ds.spec.var_names)
    except (FormulaError, ValueError):
        return None


def save(fig, stem, formats=("png", "pdf"), dpi=200):
    """Write ``stem.<fmt>`` for each format, keeping texts outside the axes."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = [stem.with_suffix(f".{f}") for f in formats]
    for path in out:
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return out


# --------------------------------------------------------------------------
# panels with one region per method
# --------------------------------------------------------------------------

def panel_style(ds, **kw):
    """Style of a panel whose regions are drawn by hand (not by draw_panel)."""
    return DEFAULT_STYLE.derive(class_labels=ds.class_labels, **PANEL_FONT).derive(**kw)


def method_style(ds, methods, widths=None):
    """A Style whose series i is the fixed appearance of ``methods[i]``;
    ``widths`` overrides the line width of some methods."""
    base, widths = DEFAULT_STYLE, widths or {}
    pick = lambda seq: [seq[METHOD_SERIES[m] % len(seq)] for m in methods]
    return base.derive(colors=pick(base.colors), markers=pick(base.markers),
                       marker_sizes=pick(base.marker_sizes), fill_alphas=pick(base.fill_alphas),
                       line_widths=[widths.get(m, w) for m, w in zip(methods, pick(base.line_widths))],
                       class_labels=ds.class_labels, label_fontsize=15, legend_fontsize=14)


_INEQ = re.compile(r"\((?:<=|>=|<|>) ")
_EQ_VAR = re.compile(r"\(= (x\d+) ")


def is_point(ds, formula):
    """True when ``formula`` fixes every feature by an equality and nothing else
    (VeriX where no feature may move): a single point in every slice."""
    return (not _INEQ.search(formula)
            and len(set(_EQ_VAR.findall(formula))) == len(ds.spec.var_names))


def draw_methods_panel(ax, ds, x, a, b, formulas, clf, title=None, widths=None,
                       resolution=300, **style_kw):
    """Background, one region per method in ``formulas`` (method -> formula),
    and the sample point.  A single-point explanation is drawn as a star."""
    stars = [m for m, f in formulas.items() if is_point(ds, f)]
    drawn = [m for m in formulas if m not in stars]
    style = method_style(ds, drawn, widths).derive(**style_kw)
    exps = [Explanation(formulas[m], METHOD_LABELS[m], m) for m in drawn]
    draw_panel(ax, Panel(point=x, axes=(a, b), explanations=exps, title=title), ds.spec,
               style=style, clf=clf, resolution=resolution)
    for m in stars:
        ax.scatter([x[a]], [x[b]], marker="*", s=12 * style.point_size,
                   color=DEFAULT_STYLE.colors[METHOD_SERIES[m]], edgecolors="black",
                   linewidths=0.6, label=METHOD_LABELS[m], zorder=99)


def draw_regions(ax, ds, polys, a, b, pin, color, marker, size, alpha, linewidth, label,
                 linestyle="-", z_fill=1, z_line=2):
    """Snapshot of ``polys`` through ``pin``: filled, outlined, vertices marked."""
    regions = regions_for_formula(polys, (a, b), ds.spec.bounds, "snapshot", point=pin)
    for n, r in enumerate(regions):
        ring = close_ring(r)
        ax.fill(ring[:, 0], ring[:, 1], color=color, alpha=alpha, zorder=z_fill)
        ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=linewidth, linestyle=linestyle,
                zorder=z_line, label=label if n == 0 else None)
        ax.scatter(r[:, 0], r[:, 1], marker=marker, s=size, color=color, zorder=z_line)
    if not regions:
        ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


# --------------------------------------------------------------------------
# where to slice an explanation that misses its sample
# --------------------------------------------------------------------------

PIN_SLACK = 1e-3  # how far past the nearest point a pin may move inwards


def _deepest(p, x, rad):
    """Point within L-inf ``rad`` of ``x`` maximising the slack of every row."""
    n = p.n_vars
    norms = np.linalg.norm(p.A_ub, axis=1)
    eye = np.eye(n)
    A = np.vstack([np.c_[p.A_ub, norms], np.c_[eye, np.zeros(n)], np.c_[-eye, np.zeros(n)]])
    b = np.r_[p.b_ub, x + rad, -(x - rad)]
    res = linprog(np.r_[np.zeros(n), -1.0], A_ub=A, b_ub=b,
                  bounds=[(0, 1)] * n + [(0, None)], method="highs")
    return (res.x[:n], float(res.x[-1])) if res.success else (None, -1.0)


def _nearest(p, x):
    """(point, L-inf distance) of the point of ``p`` closest to ``x``."""
    n = p.n_vars
    eye = np.eye(n)
    A = np.vstack([np.c_[p.A_ub, np.zeros(len(p.A_ub))],
                   np.c_[eye, -np.ones(n)], np.c_[-eye, -np.ones(n)]])
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=A, b_ub=np.r_[p.b_ub, x, -x],
                  bounds=[(0, 1)] * n + [(0, None)], method="highs")
    return (res.x[:n], float(res.x[-1])) if res.success else (None, np.inf)


def pin_point(polys, x):
    """(point, shift, degenerate): where to slice an explanation of sample ``x``.

    * contains ``x``: ``x`` itself.
    * misses ``x`` but has an interior: the deepest point within ``PIN_SLACK``
      of its nearest point, so the 2D slice has real area.
    * no interior at all: the nearest point, flagged ``degenerate`` (drawn as
      a star).
    """
    if any(p.contains(x, tol=1e-9) for p in polys):
        return x, 0.0, False
    best = (None, np.inf, True)
    for p in polys:
        near, dist = _nearest(p, x)
        if near is None:
            continue
        deep, depth = _deepest(p, x, dist + PIN_SLACK)
        if deep is not None and depth > 1e-9:
            cand = (deep, float(np.max(np.abs(deep - x))), False)
        else:
            cand = (near, dist, True)
        if (cand[2], cand[1]) < (best[2], best[1]):
            best = cand
    return best


# --------------------------------------------------------------------------
# axes
# --------------------------------------------------------------------------

def axis_label(ds, j):
    """Axis label with the SMT variable in front: ``x1-Age``."""
    return f"x{j + 1}-{ds.spec.axis_labels[j]}"


def real_ticks(ax, ds, a, b, X, nbins=4):
    """Relabel the scaled [0,1] axes with real-world values (shape unchanged).

    Ticks sit at round real values; integer-valued features get integer ticks,
    and large ones are written in thousands.
    """
    for j, axis in ((a, ax.xaxis), (b, ax.yaxis)):
        lo, hi = ds.real_bounds[j]
        real = X[:, j] * (hi - lo) + lo
        integer = bool(np.allclose(real, np.round(real), atol=1e-6))
        vals = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10],
                           integer=integer).tick_values(lo, hi)
        vals = [v for v in vals if lo - 1e-9 <= v <= hi + 1e-9]
        axis.set_major_locator(FixedLocator([(v - lo) / (hi - lo) for v in vals]))
        axis.set_major_formatter(lambda v, _k, lo=lo, hi=hi: _tick_text(lo + v * (hi - lo), hi))


def _tick_text(value, hi):
    """'640', '6.2'; thousands as '20k' once the range reaches 10000."""
    if hi >= 10000:
        return f"{round(value / 1000, 3):g}k"
    return f"{round(value, 6):g}"
