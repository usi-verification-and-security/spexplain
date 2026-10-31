"""Geometry tests: exact halfplane intersection vs the LP support sweep."""

from __future__ import annotations

import numpy as np
import pytest

from explviz.geometry.halfplanes import (close_ring, intersect_halfplanes,
                                         polygon_area)
from explviz.geometry.polytope import Polytope
from explviz.geometry.region2d import region_2d


def box_polytope(n, lo=0.0, hi=1.0):
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([np.full(n, hi), np.full(n, -lo)])
    return Polytope(A, b, np.zeros((0, n)), np.zeros(0))


def hausdorff(P, Q):
    d1 = max(np.min(np.linalg.norm(Q - p, axis=1)) for p in P)
    d2 = max(np.min(np.linalg.norm(P - q, axis=1)) for q in Q)
    return max(d1, d2)


# --------------------------------------------------------------------- box

def test_unit_square_vertices_and_area():
    v = intersect_halfplanes(np.array([[1., 0], [-1, 0], [0, 1], [0, -1]]),
                             np.array([1., 0, 1, 0]))
    assert len(v) == 4
    assert polygon_area(v) == pytest.approx(1.0)


def test_infeasible_returns_none():
    assert intersect_halfplanes(np.array([[1., 0], [-1, 0]]),
                                np.array([0., -1])) is None


def test_degenerate_segment_returns_none():
    assert intersect_halfplanes(np.array([[1., 0], [-1, 0], [0, 1], [0, -1]]),
                                np.array([0., 0, 1, 0])) is None


def test_box_only_clipping():
    v = intersect_halfplanes(np.empty((0, 2)), np.empty(0), box=[(0, 2), (0, 3)])
    assert polygon_area(v) == pytest.approx(6.0)


def test_close_ring_repeats_first_vertex():
    v = np.array([[0., 0], [1, 0], [1, 1]])
    r = close_ring(v)
    assert len(r) == 4 and np.allclose(r[0], r[-1])


# ------------------------------------------------------------- substitute

def test_substitute_matches_equality_rows():
    """Folding pinned variables in must agree with adding equality rows."""
    rng = np.random.default_rng(0)
    n = 6
    for _ in range(20):
        A = rng.normal(size=(15, n))
        b = np.abs(rng.normal(size=15)) + 3.0   # keep the origin interior
        poly = Polytope(A, b, np.zeros((0, n)), np.zeros(0))
        point = rng.uniform(-0.2, 0.2, n)
        axes = (0, 3)
        bounds = [(-5.0, 5.0)] * n

        exact = region_2d(poly, axes, bounds, "snapshot", point=point)

        # reference: pin via equality rows and project with LPs
        eq = []
        for i in range(n):
            if i not in axes:
                r = np.zeros(n); r[i] = 1.0
                eq.append((r, point[i]))
        ref_poly = Polytope(A, b, np.array([r for r, _ in eq]),
                            np.array([v for _, v in eq]))
        ref = region_2d(ref_poly, axes, bounds, "projection", n_angles=720)

        if exact is None or ref is None:
            assert exact is None and ref is None
            continue
        assert hausdorff(exact, ref) < 1e-6
        assert polygon_area(exact) == pytest.approx(polygon_area(ref), rel=1e-6)


def test_substitute_is_noop_for_absent_vars():
    p = box_polytope(3)
    assert p.substitute({}).A_ub.shape == p.A_ub.shape


def test_substitute_rejects_bad_index():
    with pytest.raises(IndexError):
        box_polytope(3).substitute({7: 0.0})


# ------------------------------------------------------ snapshot vs sweep

def test_snapshot_matches_lp_sweep_on_random_2d():
    rng = np.random.default_rng(1)
    for _ in range(25):
        A = rng.normal(size=(10, 2))
        b = np.abs(rng.normal(size=10)) + 2.0
        poly = Polytope(A, b, np.zeros((0, 2)), np.zeros(0))
        bounds = [(-5.0, 5.0)] * 2
        exact = region_2d(poly, (0, 1), bounds, "snapshot", point=np.zeros(2))
        sweep = region_2d(poly, (0, 1), bounds, "projection", n_angles=2000)
        assert exact is not None and sweep is not None
        assert polygon_area(exact) == pytest.approx(polygon_area(sweep), rel=1e-3)


def test_axis_order_is_respected():
    """Plotting (b, a) must be the mirror of plotting (a, b)."""
    n = 4
    poly = box_polytope(n)
    poly.A_ub = np.vstack([poly.A_ub, np.array([1.0, 2.0, 0, 0])])
    poly.b_ub = np.concatenate([poly.b_ub, [1.2]])
    bounds = [(0.0, 1.0)] * n
    pt = np.full(n, 0.3)
    ab = region_2d(poly, (0, 1), bounds, "snapshot", point=pt)
    ba = region_2d(poly, (1, 0), bounds, "snapshot", point=pt)
    assert polygon_area(ab) == pytest.approx(polygon_area(ba))
    assert hausdorff(ab, ba[:, ::-1]) < 1e-9


def test_empty_region_returns_none_in_both_modes():
    n = 3
    A = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    b = np.array([0.0, -1.0])           # x0 <= 0 and x0 >= 1
    poly = Polytope(A, b, np.zeros((0, n)), np.zeros(0))
    bounds = [(0.0, 1.0)] * n
    assert region_2d(poly, (0, 1), bounds, "snapshot", point=np.zeros(n)) is None
    assert region_2d(poly, (0, 1), bounds, "projection") is None


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="snapshot"):
        region_2d(box_polytope(2), (0, 1), [(0, 1)] * 2, "nonsense")
