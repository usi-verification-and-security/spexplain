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
