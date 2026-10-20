"""A convex polytope as ``A_ub x <= b_ub`` together with ``A_eq x == b_eq``.

Row convention
--------------
The SMT2 interpreter produces, for each atom, a row ``r`` of length ``n + 1``
whose last entry is the constant term, encoding ``r . (x, 1) <= 0``.
:func:`from_rows` performs the one conversion to ``A x <= b`` once, here, so
that no downstream code has to repeat the unexplained sign flip that used to
live in ``projection.project_multi_subspace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np


def _smt_number(x: float) -> str:
    """Exact SMT2 literal for a float, in the ``(/ (- p) q)`` style of the
    explanation files.  ``Fraction`` gives the float's exact binary value, so
    nothing is rounded on the way out."""
    if not np.isfinite(x):
        raise ValueError(f"cannot write non-finite number {x!r} as SMT2")
    f = Fraction(float(x))
    num = str(abs(f.numerator)) if f.numerator >= 0 else f"(- {abs(f.numerator)})"
    return num if f.denominator == 1 else f"(/ {num} {f.denominator})"


def _smt_linear(coeffs: np.ndarray, var_names: Sequence[str]) -> str:
    terms = [name if c == 1.0 else f"(* {name} {_smt_number(c)})"
             for c, name in zip(coeffs, var_names) if c != 0.0]
    if not terms:
        return "0"
    return terms[0] if len(terms) == 1 else f"(+ {' '.join(terms)})"


@dataclass
class Polytope:
    A_ub: np.ndarray  # (m, n)
    b_ub: np.ndarray  # (m,)
    A_eq: np.ndarray  # (k, n)
    b_eq: np.ndarray  # (k,)

    def __post_init__(self):
        self.A_ub = np.atleast_2d(np.asarray(self.A_ub, dtype=float))
        self.A_eq = np.atleast_2d(np.asarray(self.A_eq, dtype=float))
        self.b_ub = np.asarray(self.b_ub, dtype=float).ravel()
        self.b_eq = np.asarray(self.b_eq, dtype=float).ravel()
        if self.A_ub.size == 0:
            self.A_ub = self.A_ub.reshape(0, self.n_vars)
        if self.A_eq.size == 0:
            self.A_eq = self.A_eq.reshape(0, self.n_vars)
        if self.A_ub.shape[0] != self.b_ub.shape[0]:
            raise ValueError("A_ub and b_ub disagree on the number of rows")
        if self.A_eq.shape[0] != self.b_eq.shape[0]:
            raise ValueError("A_eq and b_eq disagree on the number of rows")
        if self.A_eq.shape[0] and self.A_eq.shape[1] != self.A_ub.shape[1]:
            raise ValueError("A_ub and A_eq disagree on the number of variables")

    @property
    def n_vars(self) -> int:
        if self.A_ub.ndim == 2 and self.A_ub.shape[1]:
            return self.A_ub.shape[1]
        if self.A_eq.ndim == 2 and self.A_eq.shape[1]:
            return self.A_eq.shape[1]
        return 0

    @classmethod
    def from_rows(cls, ineq_rows: np.ndarray, eq_rows: np.ndarray) -> "Polytope":
        """Build from ``(m, n+1)`` interpreter rows meaning ``r . (x, 1) <= 0``."""
        ineq_rows = np.atleast_2d(np.asarray(ineq_rows, dtype=float))
        eq_rows = np.atleast_2d(np.asarray(eq_rows, dtype=float))
        return cls(
            A_ub=ineq_rows[:, :-1],
            b_ub=-ineq_rows[:, -1],
            A_eq=eq_rows[:, :-1],
            b_eq=-eq_rows[:, -1],
        )

    def substitute(self, fixed: dict) -> "Polytope":
        """Pin the given variables and drop their columns.

        For ``A x <= b`` split into free columns ``F`` and pinned columns ``K``
        with values ``v``:  ``A_F x_F <= b - A_K v``.

        This is the honest way to take a 2D slice.  The old code instead
        appended an equality row per pinned variable and kept the problem
        n-dimensional, which left ``linprog`` solving a degenerate system.
        """
        if not fixed:
            return Polytope(self.A_ub.copy(), self.b_ub.copy(),
                            self.A_eq.copy(), self.b_eq.copy())

        n = self.n_vars
        bad = [i for i in fixed if not 0 <= i < n]
        if bad:
            raise IndexError(f"cannot fix variables {bad}: polytope has {n} variables")

        keep = np.array([i for i in range(n) if i not in fixed], dtype=int)
        pin = np.array(sorted(fixed), dtype=int)
        vals = np.array([fixed[i] for i in pin], dtype=float)

        return Polytope(
            A_ub=self.A_ub[:, keep],
            b_ub=self.b_ub - self.A_ub[:, pin] @ vals,
            A_eq=self.A_eq[:, keep],
            b_eq=self.b_eq - self.A_eq[:, pin] @ vals,
        )

    def to_smt2(self, var_names: Sequence[str]) -> str:
        """Write the polytope as an SMT2 conjunction ``(and (<= ...) (= ...))``.

        ``var_names[j]`` names column ``j``; after :meth:`substitute` pass the
        names of the kept variables (in increasing index order), not all of
        them.  Rows whose coefficients are all zero are constant: true ones are
        dropped, false ones are kept as ``(<= 0 b)`` so the formula stays
        unsatisfiable.  The result reads back with ``parse_formula``.
        """
        if len(var_names) != self.n_vars:
            raise ValueError(f"got {len(var_names)} names for {self.n_vars} variables")
        atoms = []
        for op, A, b in (("<=", self.A_ub, self.b_ub), ("=", self.A_eq, self.b_eq)):
            for row, rhs in zip(A, b):
                if not np.any(row):
                    holds = 0.0 <= rhs if op == "<=" else rhs == 0.0
                    if holds:
                        continue
                atoms.append(f"({op} {_smt_linear(row, var_names)} {_smt_number(rhs)})")
        return f"(and {' '.join(atoms)})" if atoms else "(and)"

    def with_box(self, bounds) -> "Polytope":
        """Return a copy with the box constraints folded in as extra rows."""
        n = self.n_vars
        rows, rhs = [], []
        for i, (lo, hi) in enumerate(bounds):
            if hi is not None and np.isfinite(hi):
                r = np.zeros(n); r[i] = 1.0
                rows.append(r); rhs.append(float(hi))
            if lo is not None and np.isfinite(lo):
                r = np.zeros(n); r[i] = -1.0
                rows.append(r); rhs.append(-float(lo))
        if not rows:
            return self
        return Polytope(
            A_ub=np.vstack([self.A_ub, np.array(rows)]),
            b_ub=np.concatenate([self.b_ub, np.array(rhs)]),
            A_eq=self.A_eq,
            b_eq=self.b_eq,
        )

    def contains(self, x, tol: float = 1e-9) -> bool:
        x = np.asarray(x, dtype=float).ravel()
        if self.A_ub.shape[0] and np.any(self.A_ub @ x - self.b_ub > tol):
            return False
        if self.A_eq.shape[0] and np.any(np.abs(self.A_eq @ x - self.b_eq) > tol):
            return False
        return True
