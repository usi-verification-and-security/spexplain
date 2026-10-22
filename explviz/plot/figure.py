"""Compose several panels into one figure with a single shared legend."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from ..style import DEFAULT_STYLE
from .legend import build_legend
from .panel import draw_panel, draw_panel_no_warnings

import numpy as np

"""
def plot_grid_scaled(panels, spec, ncols: int = 3, style=DEFAULT_STYLE, clf=None,
              shared_legend: bool = True, legend_order=None,
              figsize=None, panel_size: float = 5.0, **kw):
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
        draw_panel_no_warnings(ax, panel, spec, style=style, clf=clf, **kw)
    for ax in flat[len(panels):]:
        ax.axis("off")

    # ------------------------------------------------------ #
    # 3. RISCALAMENTO COMPLETO CORRETTO (Dati + Assi)
    # ------------------------------------------------------ #
    xmin, xmax = 100.0, 300.0
    ymin, ymax = 100.0, 300.0
    
    apply_x = (xmin is not None) and (xmax is not None)
    apply_y = (ymin is not None) and (ymax is not None)

    if apply_x or apply_y:
        x_delta = xmax - xmin if apply_x else 1.0
        y_delta = ymax - ymin if apply_y else 1.0

        for ax in flat[: len(panels)]:
            
            # A. Riscala linee standard (plt.plot)
            for line in ax.get_lines():
                if apply_x:
                    line.set_xdata(line.get_xdata() * x_delta + xmin)
                if apply_y:
                    line.set_ydata(line.get_ydata() * y_delta + ymin)
            
            # B. Riscala Collezioni (plt.scatter, plt.fill_between)
            for collection in ax.collections:
                # Gestione Scatter Plot (Offsets)
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    new_offsets = np.array(offsets, dtype=float)
                    print(new_offsets)
                    if apply_x:
                        new_offsets[:, 0] = new_offsets[:, 0] * x_delta + xmin
                    if apply_y:
                        new_offsets[:, 1] = new_offsets[:, 1] * y_delta + ymin
                    collection.set_offsets(new_offsets)
                    
                    # Aggiorna manualmente il dataLim per gli offset degli scatter
                    ax.dataLim.update_from_data_xy(new_offsets, ignore=False)
                   
                # Gestione Aree/Poligoni (Path) - es. fill_between
                from matplotlib.collections import PathCollection
                if not isinstance(collection, PathCollection):
                    paths = collection.get_paths()
                    if paths:
                        new_paths = []
                        for path in paths:
                            vertices = np.array(path.vertices, dtype=float)
                            if apply_x:
                                vertices[:, 0] = vertices[:, 0] * x_delta + xmin
                            if apply_y:
                                vertices[:, 1] = vertices[:, 1] * y_delta + ymin
                            
                            from matplotlib.path import Path
                            new_path = Path(vertices, path.codes)
                            new_paths.append(new_path)
                            
                            # CRITICO: Aggiorna la bounding box dell'asse con i nuovi vertici del path
                            ax.dataLim.update_from_path(new_path, ignore=False)
                            
                        collection.set_paths(new_paths)
                
                collection.stale = True


            # C. Riscala Rettangoli e Patch (plt.bar, istogrammi)
            for patch in ax.patches:
                from matplotlib.patches import Rectangle
                if isinstance(patch, Rectangle):
                    # Calcola la trasformazione sui punti cardine del rettangolo
                    if apply_x:
                        x_new = patch.get_x() * x_delta + xmin
                        w_new = patch.get_width() * x_delta
                        patch.set_x(x_new)
                        patch.set_width(w_new)
                    if apply_y:
                        y_new = patch.get_y() * y_delta + ymin
                        h_new = patch.get_height() * y_delta
                        patch.set_y(y_new)
                        patch.set_height(h_new)
                else:
                    # Per poligoni generici, aggiorna i vertici in modo sicuro
                    path = patch.get_path()
                    vertices = np.array(path.vertices, dtype=float)
                    if apply_x:
                        vertices[:, 0] = vertices[:, 0] * x_delta + xmin
                    if apply_y:
                        vertices[:, 1] = vertices[:, 1] * y_delta + ymin
                    path.vertices = vertices
                patch.stale = True

            # D. Forza l'aggiornamento manuale del bounding box dell'asse
            # (Risolve il problema di relim() che ignora le collections)
            ax.relim()
            
            # Se ci sono solo collezioni (scatter), aggiorniamo i limiti manualmente
            if apply_x:
                ax.set_xlim(xmin, xmax)
            else:
                ax.autoscale_view(scalex=True, scaley=False)
                
            if apply_y:
                ax.set_ylim(ymin, ymax)
            else:
                ax.autoscale_view(scalex=False, scaley=True)

    # ------------------------------------------------------- #

    if shared_legend:
        build_legend(fig, flat[: len(panels)], order=legend_order,
                     fontsize=style.legend_fontsize)
    else:
        fig.tight_layout()

    return fig
"""

def plot_grid(panels, x_min, x_max, y_min, y_max, spec, ncols: int = 3, style=DEFAULT_STYLE, clf=None,
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
        draw_panel_no_warnings(ax, panel, spec, style=style, clf=clf, **kw)
        
        # --- MODIFICA DEI TICK ---
        # Recupera i limiti reali dal pannello (adatta i nomi degli attributi se differiscono)
        xmin = getattr(panel, 'xmin', x_min)
        xmax = getattr(panel, 'xmax', x_max)
        ymin = getattr(panel, 'ymin', y_min)
        ymax = getattr(panel, 'ymax', y_max)
        
        # Mappa i punti 0 e 1 ai valori reali xmin, xmax, ymin, ymax
        ax.set_xticks([0, 1], [f"{xmin}", f"{xmax}"])
        ax.set_yticks([0, 1], [f"{ymin}", f"{ymax}"])
        # -------------------------

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
