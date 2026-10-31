"""Visual style: one object instead of a dozen parallel magic lists.

The old scripts carried per-series ``colors``/``shapes``/``sizes``/``alphas``/
``line_widths`` lists indexed positionally, which silently ran out or wrapped
when there were more series than entries.  Here a :class:`SeriesStyle` is
produced for any index by cycling, and the cycles are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence


@dataclass(frozen=True)
class SeriesStyle:
    """Resolved appearance for one explanation series."""

    color: str
    marker: str
    size: float
    alpha: float
    linewidth: float


@dataclass
class Style:
    """Appearance of a whole figure."""

    # --- explanation regions (cycled per series) ---
    colors: Sequence[str] = ("red", "darkorange", "green", "blue", "purple", "cyan")
    markers: Sequence[str] = ("o", "s", "o", "v", "^", "o")
    marker_sizes: Sequence[float] = (16, 12, 10, 10, 10, 10)
    fill_alphas: Sequence[float] = (0.20, 0.25, 0.35, 0.15, 0.30, 0.70)
    line_widths: Sequence[float] = (5, 3, 2, 1, 2, 4)

    # --- class-region background ---
    class_colors: Sequence[str] = ("orange", "lightblue")
    class_labels: Sequence[str] = ("High risk", "Low risk")
    class_markers: Sequence[str] = ("o", "s")
    contour_alpha: float = 0.25
    boundary_color: str = "black"
    boundary_width: float = 1.2
    scatter_alpha: float = 0.15
    scatter_size: float = 40.0

    # --- sample point ---
    point_color: str = "black"
    point_marker: str = "d"
    point_size: float = 40.0
    point_label: str = "Sample point"

    # --- text ---
    label_fontsize: float = 18
    tick_fontsize: float = 12
    legend_fontsize: float = 18
    usetex: bool = False

    # --- axes ---
    grid_alpha: float = 0.3
    show_spines: bool = False
    margin_frac: float = 0.05

    #: Maps raw method keys to display labels.  Unknown keys fall through
    #: unchanged rather than raising, unlike the old ``rename_single_methods``.
    label_map: dict = field(
        default_factory=lambda: {
            "naive": "Direct",
            "fix-all": "Fix-all",
            "prefer-all": "Guide-all",
            "fix-guided": "Hybrid",
            "verix": "VeriX",
        }
    )

    def series(self, i: int) -> SeriesStyle:
        """Appearance for series ``i``, cycling through each list."""
        return SeriesStyle(
            color=self.colors[i % len(self.colors)],
            marker=self.markers[i % len(self.markers)],
            size=self.marker_sizes[i % len(self.marker_sizes)],
            alpha=self.fill_alphas[i % len(self.fill_alphas)],
            linewidth=self.line_widths[i % len(self.line_widths)],
        )

    def display_label(self, key: str) -> str:
        return self.label_map.get(key, key)

    def derive(self, **kw) -> "Style":
        """A copy with some fields overridden."""
        return replace(self, **kw)

    def apply_rcparams(self) -> None:
        """Opt in to LaTeX text rendering; off by default so this works
        on machines with no LaTeX install."""
        if self.usetex:
            import matplotlib.pyplot as plt

            plt.rcParams["text.usetex"] = True


DEFAULT_STYLE = Style()
