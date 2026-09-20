"""Parse SMT2 explanation formulas into polytopes.

The formulas are lisp-style conjunctions of linear inequalities over
``x1..xn`` with exact rational coefficients, e.g.::

    (and (<= (/ 5605 16209) (+ x1 (* x2 (/ 405005 259344)) ...)) (not (<= ...)))

Each atom becomes a row ``r`` of length ``n+1`` (last entry the constant)
meaning ``r . (x, 1) <= 0``; :meth:`Polytope.from_rows` converts to ``A x <= b``.

Ported from the original ``parsing/interpret_ineq.py``.  The recursive term
interpretation is unchanged; what changed is noted inline.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pyparsing as pp

from ..geometry.polytope import Polytope


class FormulaError(ValueError):
    """Raised when a formula cannot be interpreted as a linear system."""


# --------------------------------------------------------------------------
# grammar
# --------------------------------------------------------------------------

def _make_parser():
    LP = pp.Literal("(").suppress()
    RP = pp.Literal(")").suppress()
    atom = pp.Word(pp.alphanums + "_<>=.+-*/")
    sexpr = pp.Forward()
    sexpr <<= LP + pp.Group(pp.ZeroOrMore(sexpr | atom)) + RP
    return sexpr


_PARSER = _make_parser()


def parse_sexpr(text: str) -> list:
    """Parse a lisp-style expression into nested Python lists."""
    return _PARSER.parseString(text, parseAll=True).asList()


# --------------------------------------------------------------------------
# interpretation
# --------------------------------------------------------------------------

def _as_number(token: str) -> float:
    """Numeric value of a leaf token.

    The original used ``int(token)``, so any decimal literal (``3.5``) raised
    ValueError.  Integers still parse exactly; decimals now work too.
    """
    try:
        return float(int(token))
    except ValueError:
        return float(token)


def _is_const(term: np.ndarray) -> bool:
    return bool(np.all(term[:-1] == 0.0))


def _interpret_linear_term(node, var_names: Sequence[str]) -> np.ndarray:
    """Interpret a linear term as coefficients, constant term last."""
    n = len(var_names)

    if not isinstance(node, list):
        res = np.zeros(n + 1)
        try:
            res[list(var_names).index(node)] = 1.0
        except ValueError:
            res[-1] = _as_number(node)
        return res

    if not node:
        raise FormulaError("empty sub-expression")

    op = node[0]

    if op == "ite":
        # The original computed the branches then fell through without
        # returning, silently producing nonsense.  No formula in this repo
        # uses `ite`; fail loudly if one ever does.
        raise FormulaError(
            f"'ite' is not supported in a linear term: {node!r}. "
            "Eliminate it before parsing."
        )

    term1 = _interpret_linear_term(node[1], var_names)

    if op == "+":
        for sub in node[2:]:
            term1 = term1 + _interpret_linear_term(sub, var_names)
        return term1

    if len(node) not in (2, 3):
        raise FormulaError(f"operator {op!r} with {len(node) - 1} arguments: {node!r}")

    term2 = None if len(node) == 2 else _interpret_linear_term(node[2], var_names)

    if op == "-":
        return -term1 if term2 is None else term1 - term2

    if op == "*":
        if _is_const(term1):
            return term2 * term1[-1]
        if _is_const(term2):
            return term1 * term2[-1]
        raise FormulaError(f"nonlinear product: {node!r}")

    if op == "/":
        if term2 is not None and _is_const(term2):
            if term2[-1] == 0.0:
                raise FormulaError(f"division by zero: {node!r}")
            return term1 / term2[-1]
        raise FormulaError(f"nonlinear division: {node!r}")

    raise FormulaError(f"cannot interpret linear term {node!r}")


def _interpret_atom(node, var_names: Sequence[str]) -> np.ndarray:
    """Interpret one relational atom as a row meaning ``r . (x, 1) <= 0``."""
    if len(node) == 2:
        if node[0] != "not":
            raise FormulaError(f"unary operator {node[0]!r} is not 'not': {node!r}")
        # not (a <= b)  ==>  a > b  ==>  b - a <= 0 (closure)
        return -_interpret_atom(node[1], var_names)

    if len(node) == 4 and node[0] in ("<", ">", "!") and node[1] == "=":
        # e.g. "(>   = x4 ...)" tokenizes as ['>', '=', 'x4', ...]
        raise FormulaError(
            f"operator {node[0] + node[1]!r} is split by whitespace in the source "
            f"formula: {node[:2]!r}. Pass repair=True to parse_formula(), or fix "
            "the .phi.txt line."
        )

    if len(node) != 3:
        raise FormulaError(f"expected a binary relation, got {node!r}")

    op, lhs, rhs = node
    if op == ">=":
        op, lhs, rhs = "<=", rhs, lhs
    if op not in ("<=", "="):
        raise FormulaError(f"unsupported relation {op!r} in {node!r}")

    # t1 <= t2  ==>  t1 - t2 <= 0
    return _interpret_linear_term(lhs, var_names) - _interpret_linear_term(rhs, var_names)


def _interpret_conjunction(node, var_names: Sequence[str]):
    """Split a conjunction into inequality rows and equality rows."""
    if node[0] == "not":
        if len(node) != 2:
            raise FormulaError(f"'not' with {len(node) - 1} arguments: {node!r}")
        return [_interpret_atom(node, var_names)], []

    if node[0] != "and":
        raise FormulaError(f"expected 'and' or 'not' at the top, got {node[0]!r}")

    ineqs, eqs = [], []
    for child in node[1:]:
        if not isinstance(child, list) or not child:
            raise FormulaError(f"unexpected conjunct {child!r}")
        if child[0] == "and":
            sub_i, sub_e = _interpret_conjunction(child, var_names)
            ineqs.extend(sub_i)
            eqs.extend(sub_e)
        elif child[0] == "=":
            eqs.append(_interpret_atom(child, var_names))
        else:
            ineqs.append(_interpret_atom(child, var_names))
    return ineqs, eqs


def _stack(rows, n: int) -> np.ndarray:
    return np.stack(rows, 0) if rows else np.empty((0, n + 1))


_SPLIT_OP = re.compile(r"\(\s*([<>!])\s+=\s")


def repair_split_operators(text: str) -> str:
    """Rejoin relational operators that a producer split with whitespace.

    Some generated ``.phi.txt`` lines contain ``(>   = x4 ...)`` instead of
    ``(>= x4 ...)``.  This is a defect in the source file; repairing it is
    opt-in so that corrupt input is never silently reinterpreted.
    """
    return _SPLIT_OP.sub(r"(\1= ", text)


def parse_formula(text: str, var_names: Sequence[str], repair: bool = False) -> list:
    """Parse one formula into a list of :class:`Polytope`.

    A top-level ``(or ...)`` yields one polytope per disjunct (a union); any
    other formula yields a single-element list.

    Set ``repair=True`` to rejoin whitespace-split relational operators
    (see :func:`repair_split_operators`).
    """
    if repair:
        text = repair_split_operators(text)
    parsed = parse_sexpr(text)
    if len(parsed) != 1:
        raise FormulaError(f"expected one top-level expression, got {len(parsed)}")
    node = parsed[0]
    n = len(var_names)

    if node[0] == "or":
        disjuncts = []
        for child in node[1:]:
            ineqs, eqs = _interpret_conjunction(child, var_names)
            disjuncts.append(Polytope.from_rows(_stack(ineqs, n), _stack(eqs, n)))
        return disjuncts

    ineqs, eqs = _interpret_conjunction(node, var_names)
    return [Polytope.from_rows(_stack(ineqs, n), _stack(eqs, n))]


def variables_in(text: str) -> list:
    """Every ``x``-prefixed atom occurring in a formula (unordered)."""
    def walk(node, out):
        if isinstance(node, list):
            for child in node:
                walk(child, out)
        elif isinstance(node, str) and node.startswith("x"):
            out.add(node)

    found = set()
    walk(parse_sexpr(text), found)
    return sorted(found, key=lambda v: int(v[1:]) if v[1:].isdigit() else 0)
