"""Core data types: what to plot, and about which dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Mode = Literal["snapshot", "projection"]
Background = Literal["auto", "contour", "scatter", "none"]


def resolve_background(mode: str, background: str) -> str:
    """Which background to actually draw, given the mode.

    In ``projection`` mode there is **never** a background.  The class regions
    are the model evaluated on the plane *through the sample point* -- a slice.
    A projection is the shadow of the whole polytope onto that plane, so the
    two describe different things and drawing them together is misleading.

    ``"auto"`` therefore means ``"contour"`` for a snapshot and ``"none"`` for
    a projection.
    """
    if mode == "projection":
        return "none"
    if background == "auto":
        return "contour"
    return background


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the plotter needs to know about an input space.

    Replaces the module-level globals in the old ``Spec.py`` and the
    ``bounds = [(0, 1) for _ in range(len(lower_bounds))]`` idiom that silently
    depended on a star-import.
    """

    name: str
    var_names: list[str]
    axis_labels: list[str]
    bounds: list[tuple[float, float]]
    scaled: bool = False

    def __post_init__(self):
        n = len(self.var_names)
        if not (len(self.axis_labels) == len(self.bounds) == n):
            raise ValueError(
                f"{self.name}: var_names ({n}), axis_labels "
                f"({len(self.axis_labels)}) and bounds ({len(self.bounds)}) "
                "must all have the same length"
            )

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    def label(self, i: int) -> str:
        return self.axis_labels[i]


@dataclass
class Explanation:
    """One explanation formula, as a single line of a ``.phi.txt`` file."""

    formula: str
    label: str
    key: str | None = None


@dataclass
class Panel:
    """One subplot: a sample point, an axis pair, and the explanations to draw."""

    point: np.ndarray
    axes: tuple[int, int]
    explanations: list[Explanation] = field(default_factory=list)
    mode: Mode = "snapshot"
    background: Background = "auto"
    title: str | None = None

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=float).ravel()
        a, b = self.axes
        if a == b:
            raise ValueError(f"axes must be two distinct features, got {self.axes}")

    @property
    def free_dims(self) -> tuple[int, int]:
        """The two feature indices that stay free; every other one is pinned."""
        return self.axes

    def effective_background(self) -> str:
        """The background actually drawn, after :func:`resolve_background`."""
        return resolve_background(self.mode, self.background)

    def fixed_dims(self, n_vars: int) -> dict[int, float]:
        """Map of pinned feature index -> the sample point's value there."""
        a, b = self.axes
        return {i: float(self.point[i]) for i in range(n_vars) if i not in (a, b)}


def explanations_from_formulas(
    formulas: Sequence[str],
    keys: Sequence[str],
    label_map: dict[str, str] | None = None,
) -> list[Explanation]:
    """Zip raw formula lines with their method keys into Explanation objects."""
    if len(formulas) != len(keys):
        raise ValueError(f"got {len(formulas)} formulas but {len(keys)} keys")
    label_map = label_map or {}
    return [
        Explanation(formula=f, label=label_map.get(k, k), key=k)
        for f, k in zip(formulas, keys)
    ]
