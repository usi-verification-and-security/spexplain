"""Sanity checks that catch silently-wrong explanation data.

A snapshot is taken *through* the sample point, so an explanation that does
not contain its own sample point yields an empty region -- which on a plot
looks identical to "this method produced nothing".  Checking explicitly turns
that into a diagnosable condition.
"""

from __future__ import annotations

import numpy as np


def violated_constraints(poly, point, tol: float = 1e-9) -> np.ndarray:
    """Indices of inequality rows that ``point`` violates."""
    point = np.asarray(point, dtype=float).ravel()[: poly.n_vars]
    if poly.A_ub.shape[0] == 0:
        return np.empty(0, dtype=int)
    return np.nonzero(poly.A_ub @ point - poly.b_ub > tol)[0]


def contains_point(poly, point, tol: float = 1e-9) -> bool:
    """Whether the explanation actually contains the sample it explains."""
    return len(violated_constraints(poly, point, tol)) == 0


def describe_violations(poly, point, var_names=None, limit: int = 5) -> str:
    """Human-readable summary of why ``point`` falls outside ``poly``."""
    rows = violated_constraints(poly, point)
    if len(rows) == 0:
        return "point satisfies the explanation"
    point = np.asarray(point, dtype=float).ravel()
    names = list(var_names) if var_names else [f"x{i+1}" for i in range(poly.n_vars)]
    parts = []
    for r in rows[:limit]:
        nz = np.nonzero(np.abs(poly.A_ub[r]) > 1e-12)[0]
        lhs = poly.A_ub[r] @ point[: poly.n_vars]
        if len(nz) == 1:
            i = nz[0]
            parts.append(f"{names[i]}={point[i]:.6g} breaks "
                         f"{poly.A_ub[r][i]:+g}*{names[i]} <= {poly.b_ub[r]:.6g}")
        else:
            parts.append(f"row {r}: {lhs:.6g} > {poly.b_ub[r]:.6g}")
    extra = "" if len(rows) <= limit else f" (+{len(rows) - limit} more)"
    return f"{len(rows)} violated constraint(s): " + "; ".join(parts) + extra


def find_containing_samples(poly, X, tol: float = 1e-9) -> list:
    """Which rows of ``X`` the explanation contains.

    Useful for auditing an index map: an explanation should contain exactly
    the sample it was generated for.
    """
    return [i for i in range(len(X)) if contains_point(poly, X[i], tol)]
