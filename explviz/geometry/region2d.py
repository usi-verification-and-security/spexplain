"""Reduce an n-dimensional explanation polytope to a 2D region to draw.

Two modes, which the old code conflated behind an inverted flag named
``is_partitioned``:

``snapshot``
    Pin every non-plotted variable to the sample point's coordinate and draw
    the resulting 2D slice.  Computed exactly by substitution + halfplane
    intersection.

``projection``
    Draw the shadow of the full polytope on the chosen plane -- the set of
    (x_a, x_b) for which *some* completion satisfies the explanation.
    Computed by an LP support sweep, then convex-hulled.

Because substituting a variable that does not occur in a formula is a no-op,
explanations precomputed for a single axis pair work under either mode, and
the ``is_partitioned`` bookkeeping disappears.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from .halfplanes import intersect_halfplanes, polygon_area
from .polytope import Polytope

DEFAULT_N_ANGLES = 360


def region_2d(
    poly: Polytope,
    axes,
    bounds,
    mode: str = "snapshot",
    point=None,
    n_angles: int = DEFAULT_N_ANGLES,
):
    """2D region of ``poly`` on the plane spanned by ``axes``.

    Returns an ordered (k, 2) vertex ring, or ``None`` if the region is empty.
    """
    if mode == "snapshot":
        return _snapshot(poly, axes, bounds, point)
    if mode == "projection":
        return _projection(poly, axes, bounds, n_angles)
    raise ValueError(f"mode must be 'snapshot' or 'projection', got {mode!r}")


def _snapshot(poly: Polytope, axes, bounds, point):
    a, b = axes
    n = poly.n_vars
    if point is None:
        raise ValueError("mode='snapshot' needs the sample point to pin the other axes")
    point = np.asarray(point, dtype=float).ravel()
    if point.size < n:
        raise ValueError(f"point has {point.size} features but the formula has {n}")

    sliced = poly.substitute({i: point[i] for i in range(n) if i not in (a, b)})

    # Equalities in two variables: fold each into a pair of inequalities.
    A, rhs = sliced.A_ub, sliced.b_ub
    if sliced.A_eq.shape[0]:
        A = np.vstack([A, sliced.A_eq, -sliced.A_eq])
        rhs = np.concatenate([rhs, sliced.b_eq, -sliced.b_eq])

    # substitute() keeps columns in increasing index order.
    box = [bounds[a], bounds[b]] if a < b else [bounds[b], bounds[a]]
    verts = intersect_halfplanes(A, rhs, box=box)
    if verts is None:
        return None
    return verts if a < b else verts[:, ::-1]


def _projection(poly: Polytope, axes, bounds, n_angles: int):
    a, b = axes
    n = poly.n_vars
    A_ub = poly.A_ub if poly.A_ub.shape[0] else None
    b_ub = poly.b_ub if poly.A_ub.shape[0] else None
    A_eq = poly.A_eq if poly.A_eq.shape[0] else None
    b_eq = poly.b_eq if poly.A_eq.shape[0] else None

    supports = []
    for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        c = np.zeros(n)
        c[a], c[b] = np.cos(theta), np.sin(theta)
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.success:
            supports.append((res.x[a], res.x[b]))

    if len(supports) < 3:
        return None
    pts = np.array(supports, dtype=float)

    # The shadow of a convex set is convex, so the hull of the sampled support
    # points is the region -- and it drops the duplicate/near-duplicate points
    # that made the old fill look jagged.
    scale = max(1.0, float(np.abs(pts).max()))
    pts = np.unique(np.round(pts / scale, 9), axis=0) * scale
    if pts.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None
    verts = pts[hull.vertices]
    return verts if polygon_area(verts) > 0 else None


def regions_for_formula(polytopes, axes, bounds, mode, point=None, n_angles=DEFAULT_N_ANGLES):
    """Apply :func:`region_2d` to each disjunct, dropping the empty ones."""
    out = []
    for poly in polytopes:
        r = region_2d(poly, axes, bounds, mode=mode, point=point, n_angles=n_angles)
        if r is not None:
            out.append(r)
    return out
