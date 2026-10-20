"""Draw one Axes: class regions, explanation regions, and the sample point.

This is the single workhorse behind both "plot one explanation" and "compare
several explanations of the same sample point" -- the former is just the
one-element case of the latter.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

from ..background import class_region_grid, class_region_samples
from ..config import Explanation, Panel
from ..geometry.halfplanes import close_ring
from ..geometry.region2d import regions_for_formula
from ..parsing.smt2 import parse_formula
from ..style import DEFAULT_STYLE
from ..validate import contains_point, describe_violations

# Parsing a 48-term formula costs ~200 ms, and the same formula is reused
# across axis pairs, so memoize it.
@lru_cache(maxsize=512)
def _parse_cached(formula: str, var_names: tuple, repair: bool):
    return parse_formula(formula, list(var_names), repair=repair)


def draw_panel(ax, panel: Panel, spec, style=DEFAULT_STYLE, clf=None,
               resolution: int = 300, num_samples: int = 5000,
               n_angles: int = 360, repair: bool = False, seed=None):
    """Render ``panel`` onto ``ax``."""
    a, b = panel.axes
    bounds = spec.bounds

    # --- background: class regions -------------------------------------
    background = panel.effective_background()

    if background == "none" and panel.background not in ("auto", "none"):
        warnings.warn(
            f"background={panel.background!r} is ignored in projection mode: the "
            "class regions are a slice through the sample point, while a "
            "projection is the shadow of the whole polytope, so the two are not "
            "comparable. Use mode='snapshot' to see the class regions.",
            RuntimeWarning, stacklevel=2,
        )

    if background != "none":
        if clf is None:
            raise ValueError(
                f"background={background!r} needs a classifier; pass clf=..."
            )
        if background == "contour":
            _draw_contour(ax, clf, panel, spec, style, resolution)
        elif background == "scatter":
            _draw_scatter(ax, clf, panel, spec, style, num_samples, seed)
        else:
            raise ValueError(f"unknown background {background!r}")

    # --- explanation regions -------------------------------------------
    for i, exp in enumerate(panel.explanations):
        series = style.series(i)
        polys = _parse_cached(exp.formula, tuple(spec.var_names), repair)
        regions = regions_for_formula(
            polys, panel.axes, bounds, panel.mode,
            point=panel.point, n_angles=n_angles,
        )
        if panel.mode == "snapshot":
            for poly in polys:
                if not contains_point(poly, panel.point):
                    warnings.warn(
                        f"explanation {exp.key or exp.label!r} does not contain its "
                        f"sample point, so its snapshot on axes {panel.axes} is empty: "
                        + describe_violations(poly, panel.point, spec.var_names),
                        RuntimeWarning, stacklevel=2,
                    )

        label = exp.label or style.display_label(exp.key or str(i))
        for k, region in enumerate(regions):
            ring = close_ring(region)
            ax.fill(ring[:, 0], ring[:, 1], color=series.color,
                    alpha=series.alpha, zorder=1)
            ax.plot(ring[:, 0], ring[:, 1], color=series.color,
                    linewidth=series.linewidth, zorder=2,
                    label=label if k == 0 else None)
            ax.scatter(region[:, 0], region[:, 1], marker=series.marker,
                       s=series.size, color=series.color, zorder=2)
        if not regions:
            # keep the legend entry even when this explanation is empty here
            ax.plot([], [], color=series.color, linewidth=series.linewidth,
                    label=label)

    # --- the sample point, on top of everything ------------------------
    ax.scatter([panel.point[a]], [panel.point[b]], marker=style.point_marker,
               color=style.point_color, s=style.point_size,
               label=style.point_label, zorder=100,
               edgecolors="white", linewidths=0.8)

    _finish_axes(ax, panel, spec, style)


def _draw_contour(ax, clf, panel, spec, style, resolution):
    a, b = panel.axes
    grid = class_region_grid(clf, panel.point, panel.axes, spec.bounds, resolution)
    ax.contourf(grid.xx, grid.yy, grid.proba, levels=[0.0, 0.5, 1.0],
                colors=list(style.class_colors), alpha=style.contour_alpha, zorder=0)
    if grid.proba.min() < 0.5 < grid.proba.max():
        ax.contour(grid.xx, grid.yy, grid.proba, levels=[0.5],
                   colors=[style.boundary_color],
                   linewidths=style.boundary_width, zorder=0)
    # proxy artists so the legend shows the two classes
    for colour, label in zip(style.class_colors, style.class_labels):
        ax.fill([], [], color=colour, alpha=style.contour_alpha * 2, label=label)


def _draw_scatter(ax, clf, panel, spec, style, num_samples, seed):
    bg = class_region_samples(clf, panel.point, panel.axes, spec.bounds,
                              num_samples, seed=seed)
    for pts, colour, marker, label in zip(
        (bg.class0, bg.class1), style.class_colors,
        style.class_markers, style.class_labels,
    ):
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], marker=marker, color=colour,
                   alpha=style.scatter_alpha, s=style.scatter_size, zorder=0)
        # a single opaque proxy point carries the legend entry
        ax.scatter(pts[:1, 0], pts[:1, 1], marker=marker, color=colour,
                   alpha=0.6, s=style.scatter_size, label=label, zorder=0)


def _finish_axes(ax, panel, spec, style):
    a, b = panel.axes
    (xlo, xhi), (ylo, yhi) = spec.bounds[a], spec.bounds[b]
    mx, my = (xhi - xlo) * style.margin_frac, (yhi - ylo) * style.margin_frac
    ax.set_xlim(xlo - mx, xhi + mx)
    ax.set_ylim(ylo - my, yhi + my)
    ax.set_xlabel(spec.label(a), fontsize=style.label_fontsize)
    ax.set_ylabel(spec.label(b), fontsize=style.label_fontsize)
    ax.tick_params(labelsize=style.tick_fontsize)
    ax.grid(alpha=style.grid_alpha)
    if panel.title:
        ax.set_title(panel.title, fontsize=style.label_fontsize)
    if not style.show_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)


# -------------------------------------------------------------------------
# convenience wrappers (module 1 and module 2 of the requested API)
# -------------------------------------------------------------------------

def plot_explanation(explanation, point, axes, spec, **kw):
    """Plot a single explanation.  Returns ``(fig, ax)``."""
    if isinstance(explanation, str):
        explanation = Explanation(formula=explanation, label="Explanation")
    return plot_comparison([explanation], point, axes, spec, **kw)


def plot_comparison(explanations, point, axes, spec, mode="snapshot",
                    background="auto", title=None, ax=None,
                    style=DEFAULT_STYLE, clf=None, figsize=(8, 8), **kw):
    """Plot several explanations of the same sample point on one Axes."""
    import matplotlib.pyplot as plt

    style.apply_rcparams()
    panel = Panel(point=point, axes=axes, explanations=list(explanations),
                  mode=mode, background=background, title=title)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    draw_panel(ax, panel, spec, style=style, clf=clf, **kw)
    return fig, ax
