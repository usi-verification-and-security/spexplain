"""explviz -- visualizing logical explanations of neural-network decisions.

An *explanation* is an SMT2 formula describing a convex subspace of a model's
input space.  This package parses such formulas, reduces them to a 2D region
(a **snapshot** through a sample point, or a **projection** of the whole
subspace), and draws them against the model's class regions.

Quick start
-----------
>>> from explviz import Classifier, Explanation, Panel, plot_grid
>>> from explviz.specs import HEART_ATTACK_SCALED as spec
>>> from explviz.datasets import load_points, load_explanations
"""

from __future__ import annotations

from .config import DatasetSpec, Explanation, Panel, explanations_from_formulas
from .datasets import load_explanations, load_points, select_explanation
from .geometry.polytope import Polytope
from .geometry.region2d import region_2d, regions_for_formula
from .models.classifier import Classifier
from .parsing.smt2 import parse_formula
from .plot import (draw_panel, plot_comparison, plot_explanation, plot_grid,
                   save_figure)
from .style import DEFAULT_STYLE, Style

__all__ = [
    "DatasetSpec", "Explanation", "Panel", "explanations_from_formulas",
    "load_points", "load_explanations", "select_explanation",
    "Polytope", "parse_formula", "region_2d", "regions_for_formula",
    "Classifier", "Style", "DEFAULT_STYLE",
    "draw_panel", "plot_explanation", "plot_comparison", "plot_grid",
    "save_figure",
]
