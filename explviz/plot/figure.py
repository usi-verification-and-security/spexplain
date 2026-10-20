"""Compose several panels into one figure with a single shared legend."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from ..style import DEFAULT_STYLE
from .legend import build_legend
from .panel import draw_panel


def plot_grid(panels, spec, ncols: int = 3, style=DEFAULT_STYLE, clf=None,
              shared_legend: bool = True, legend_order=None,
              figsize=None, panel_size: float = 5.0, **kw):
    """Lay out ``panels`` on a grid and return the Figure.

    Extra keyword arguments are forwarded to :func:`draw_panel` (``resolution``,
    ``num_samples``, ``n_angles``, ``repair``, ``seed``).
    """
    panels = list(panels)
    if not panels:
        raise ValueError("plot_grid needs at least one panel")

    style.apply_rcparams()
    ncols = max(1, min(ncols, len(panels)))
    nrows = math.ceil(len(panels) / ncols)

    if figsize is None:
        width = panel_size * ncols + (panel_size if shared_legend else 0)
        figsize = (width, panel_size * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.reshape(-1)

    for ax, panel in zip(flat, panels):
        draw_panel(ax, panel, spec, style=style, clf=clf, **kw)
    for ax in flat[len(panels):]:
        ax.axis("off")

    if shared_legend:
        build_legend(fig, flat[: len(panels)], order=legend_order,
                     fontsize=style.legend_fontsize)
    else:
        fig.tight_layout()

    return fig


def save_figure(fig, path, formats=("png", "pdf"), dpi: int = 200):
    """Save ``fig`` under ``path`` (without suffix) in several formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    extra = [a for a in fig.axes if not a.get_visible() or a.get_legend()]
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight",
                    bbox_extra_artists=[a.get_legend() for a in extra if a.get_legend()],
                    dpi=dpi)
        written.append(out)
    return written
