"""Parser tests, including the vectors that were commented out in the original."""

from __future__ import annotations

import numpy as np
import pytest

from explviz.parsing.smt2 import (FormulaError, parse_formula,
                                  repair_split_operators, variables_in)
from explviz.parsing.smt2 import _interpret_linear_term as term

V = ["x1", "x2", "x3"]


def test_linear_term_constant_folding():
    np.testing.assert_almost_equal(term(["+", "3.5", "6.7"], V), [0, 0, 0, 10.2])


def test_linear_term_product_of_constants():
    np.testing.assert_almost_equal(term(["+", ["*", "4", "-3.3"], "6.7"], V),
                                   [0, 0, 0, -6.5])


def test_linear_term_mixed():
    t = ["-", ["+", ["*", ["/", "5", "2"], "x1"], ["*", "x3", "3.3"]], "6.6"]
    np.testing.assert_almost_equal(term(t, V), [2.5, 0.0, 3.3, -6.6])


def test_linear_term_bare_sum():
    np.testing.assert_almost_equal(term(["+", "x1", "x3"], V), [1, 0, 1, 0])


def test_linear_term_nested_division():
    a = ["-", ["+", ["/", "x1", "3"], ["*", "x2", "2.2"]], ["*", "5.1", "x3"]]
    b = ["+", "x1", "x3"]
    t = ["+", ["/", a, ["*", "2", "3.0"]], b]
    np.testing.assert_almost_equal(term(t, V), [19 / 18, 2.2 / 6, -5.1 / 6 + 1, 0.0])


def test_decimal_literals_parse():
    """The original used int() and raised on any decimal literal."""
    np.testing.assert_almost_equal(term("3.5", V), [0, 0, 0, 3.5])


def test_simple_box():
    p = parse_formula("(and (<= x1 1) (>= x1 0) (<= x2 1) (>= x2 0))", ["x1", "x2"])[0]
    assert p.contains([0.5, 0.5])
    assert not p.contains([1.5, 0.5])


def test_ge_is_normalized():
    a = parse_formula("(and (>= x1 2))", ["x1"])[0]
    b = parse_formula("(and (<= 2 x1))", ["x1"])[0]
    np.testing.assert_allclose(a.A_ub, b.A_ub)
    np.testing.assert_allclose(a.b_ub, b.b_ub)


def test_not_negates():
    p = parse_formula("(and (not (<= x1 1)))", ["x1"])[0]
    assert p.contains([2.0]) and not p.contains([0.0])


def test_equality_goes_to_A_eq():
    p = parse_formula("(and (= x1 2) (<= x2 1))", ["x1", "x2"])[0]
    assert p.A_eq.shape[0] == 1 and p.A_ub.shape[0] == 1


def test_or_yields_several_polytopes():
    polys = parse_formula("(or (and (<= x1 0)) (and (>= x1 5)))", ["x1"])
    assert len(polys) == 2


def test_ite_raises_clearly():
    with pytest.raises(FormulaError, match="ite"):
        parse_formula("(and (<= x1 (ite (<= x2 1) x2 x3)))", V)


def test_nonlinear_raises():
    with pytest.raises(FormulaError, match="nonlinear"):
        parse_formula("(and (<= (* x1 x2) 1))", V)


def test_split_operator_error_is_actionable():
    with pytest.raises(FormulaError, match="split by whitespace"):
        parse_formula("(and (>   = x1 1))", V)


def test_split_operator_repair():
    assert repair_split_operators("(and (>   = x1 1))") == "(and (>= x1 1))"
    p = parse_formula("(and (>   = x1 1))", V, repair=True)[0]
    assert p.contains([2, 0, 0])


def test_variables_in_is_sorted_numerically():
    assert variables_in("(and (<= x1 x12) (>= x3 2))") == ["x1", "x3", "x12"]


# ------------------------------------------------------------ slice_formula

from explviz.geometry.polytope import Polytope  # noqa: E402
from explviz.geometry.region2d import region_2d  # noqa: E402
from explviz.parsing.smt2 import slice_formula  # noqa: E402


def test_to_smt2_round_trips():
    rng = np.random.default_rng(1)
    p = Polytope(rng.normal(size=(5, 3)), rng.normal(size=5),
                 rng.normal(size=(1, 3)), rng.normal(size=1))
    (q,) = parse_formula(p.to_smt2(V), V)
    for got, want in [(q.A_ub, p.A_ub), (q.b_ub, p.b_ub),
                      (q.A_eq, p.A_eq), (q.b_eq, p.b_eq)]:
        np.testing.assert_array_equal(got, want)


def test_to_smt2_drops_true_constant_rows_keeps_false_ones():
    p = Polytope(np.array([[0., 0], [0, 0], [1, 0]]), np.array([1., -1, 2]),
                 np.zeros((0, 2)), np.zeros(0))
    assert p.to_smt2(["x1", "x2"]) == "(and (<= 0 (- 1)) (<= x1 2))"


def test_slice_formula_matches_snapshot():
    names = [f"x{i}" for i in range(1, 6)]
    formula = ("(and (<= (+ x1 (* x3 (/ 1 2)) x5) 3) (not (<= x2 (- 2)))"
               " (<= (+ (* (- 1) x4) x3) (/ 7 3)) (<= x1 (+ x2 4)))")
    point = np.array([0.5, 0.1, -0.3, 0.7, 0.2])
    bounds = [(-5.0, 5.0)] * 5
    for axes in [(0, 2), (2, 0), (1, 4)]:
        sliced = slice_formula(formula, names, point, axes)
        assert set(variables_in(sliced)) <= {names[i] for i in axes}
        (orig,) = parse_formula(formula, names)
        (back,) = parse_formula(sliced, names)
        want = region_2d(orig, axes, bounds, "snapshot", point=point)
        got = region_2d(back, axes, bounds, "snapshot", point=point)
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_slice_formula_keeps_disjuncts():
    sliced = slice_formula("(or (and (<= x1 1)) (and (<= x2 x3)))", V, [0, 0, 0.5], (0, 1))
    assert sliced == "(or (and (<= x1 1)) (and (<= x2 (/ 1 2))))"
