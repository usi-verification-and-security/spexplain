"""Exact vertex enumeration for a 2D polytope ``A x <= b``.

Once the non-plotted variables are pinned (see :meth:`Polytope.substitute`),
a snapshot is just an intersection of halfplanes in the plane.  With only a
few dozen constraints the O(m^2) pairwise-intersection approach is both fast
and robust, and it is *exact* -- unlike sweeping 1000 LP directions and hoping
the sampled support points trace the boundary smoothly.
"""

from __future__ import annotations

import numpy as np


def _normalize(A: np.ndarray, b: np.ndarray):
    """Scale each row to unit normal so tolerances are in geometric units."""
    norms = np.linalg.norm(A, axis=1)
    keep = norms > 0
    trivial = ~keep & (b < -1e-9)  # 0 . x <= negative  =>  infeasible
    if np.any(trivial):
        return None, None
    A, b, norms = A[keep], b[keep], norms[keep]
    return A / norms[:, None], b / norms


def intersect_halfplanes(A, b, box=None, tol: float = 1e-7):
    """Vertices of ``{x in R^2 : A x <= b}``, ordered counter-clockwise.

    Parameters
    ----------
    A, b : the halfplane system, shapes (m, 2) and (m,).
    box  : optional [(xlo, xhi), (ylo, yhi)] clipping rectangle.
    tol  : geometric feasibility tolerance.

    Returns the (k, 2) array of vertices without repeating the first, or
    ``None`` if the region is empty or lower-dimensional.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    if A.size == 0:
        A = np.empty((0, 2))
    if A.shape[1] != 2:
        raise ValueError(f"expected 2 variables, got {A.shape[1]}")

    if box is not None:
        (xlo, xhi), (ylo, yhi) = box
        extra_A = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
        extra_b = np.array([xhi, -xlo, yhi, -ylo], dtype=float)
        A = np.vstack([A, extra_A]) if A.shape[0] else extra_A
        b = np.concatenate([b, extra_b]) if b.size else extra_b

    A, b = _normalize(A, b)
    if A is None or A.shape[0] < 3:
        return None

    # Candidate vertices: intersections of every pair of boundary lines.
    m = A.shape[0]
    i, j = np.triu_indices(m, k=1)
    a1, a2 = A[i], A[j]
    det = a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0]
    ok = np.abs(det) > 1e-12
    if not np.any(ok):
        return None
    i, j, det = i[ok], j[ok], det[ok]
    b1, b2 = b[i], b[j]
    a1, a2 = A[i], A[j]
    px = (b1 * a2[:, 1] - b2 * a1[:, 1]) / det
    py = (a1[:, 0] * b2 - a2[:, 0] * b1) / det
    pts = np.column_stack([px, py])

    # Keep those satisfying every constraint.
    feasible = np.all(pts @ A.T <= b[None, :] + tol, axis=1)
    pts = pts[feasible]
    if pts.shape[0] < 3:
        return None

    # Deduplicate, then order counter-clockwise about the centroid.
    scale = max(1.0, float(np.abs(pts).max()))
    _, uniq = np.unique(np.round(pts / scale, 9), axis=0, return_index=True)
    pts = pts[np.sort(uniq)]
    if pts.shape[0] < 3:
        return None

    centre = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    pts = pts[order]

    if polygon_area(pts) <= tol * tol:
        return None  # degenerate: a point or a segment
    return pts


def polygon_area(pts: np.ndarray) -> float:
    """Absolute area of a simple polygon given as an ordered vertex ring."""
    if pts is None or len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def close_ring(pts: np.ndarray) -> np.ndarray:
    """Append the first vertex so matplotlib draws the closing edge."""
    if pts is None or len(pts) == 0:
        return pts
    return np.vstack([pts, pts[:1]])
