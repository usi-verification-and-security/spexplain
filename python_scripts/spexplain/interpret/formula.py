# Explanations as formulas in the SMT-LIB format, over the variables x1, ..., xN as in the engine

import re

import numpy as np

from .. import Error
from . import parse_smt2


def make_variables(n_vars):
    return ['x%d' % (n + 1) for n in range(n_vars)]


def parse_vars(args, n_vars, technique):
    """Parse the features 'x<i>' or ranges 'x<i>-x<j>' (from 1) into sorted indices (from 0)."""
    if not args:
        raise Error(f'Missing the features of parameter vars of {technique}')
    indices = set()
    for arg in args:
        m = re.fullmatch(r'x(\d+)(?:-x(\d+))?', arg)
        first, last = (int(m[1]), int(m[2] or m[1])) if m else (0, 0)
        if not 1 <= first <= last <= n_vars:
            raise Error(f'Invalid features for {n_vars} features: {arg}')
        indices.update(range(first - 1, last))
    return sorted(indices)


def parse(explanation, n_vars):
    """Parse the explanation into constraints A x <= B."""
    parsed_ineq = parse_smt2.parse_and_interpret(explanation, make_variables(n_vars))
    return -parsed_ineq[:, :-1], parsed_ineq[:, -1]


def _row_key(row, b):
    # Also identifies -0.0 with 0.0
    return (row + 0.0).tobytes() + np.float64(b + 0.0).tobytes()


def equalities(A, B):
    """The rows that form equalities with an opposite row: maps the first of each pair to the second."""
    first_of = {}
    pairs = {}
    for i, (row, b) in enumerate(zip(A, B)):
        j = first_of.pop(_row_key(-row, -b), None)
        if j is not None:
            pairs[j] = i
        else:
            first_of.setdefault(_row_key(row, b), i)
    return pairs


def substitute(A, B, fixed):
    """Substitute the values for the variables {index: value} in the constraints A x <= B.

    The columns of the fixed variables become zero, the constraints without variables are removed,
    fail if any of them is violated.
    """
    A, B = A.copy(), B.copy()
    for i, v in fixed.items():
        B -= A[:, i] * v
        A[:, i] = 0
    constant = ~A.any(axis=1)
    if np.any(B[constant] < 0):
        raise Error('The explanation is empty after the substitution')
    return A[~constant], B[~constant]


def fix(A, B, fixed):
    """Add the equalities of the variables {index: value} to the constraints A x <= B."""
    n_vars = A.shape[1]
    rows, rhs = [A], [B]
    for i, v in fixed.items():
        e = np.zeros(n_vars)
        e[i] = 1
        rows += [e[None, :], -e[None, :]]
        rhs += [[v], [-v]]
    return np.vstack(rows), np.concatenate(rhs)


def is_interval(A):
    """Whether each constraint bounds just a single variable."""
    return np.all(np.count_nonzero(A, axis=1) == 1)


def interval_bounds(A, B):
    """The lower and the upper bounds of an interval explanation, within the domains [0,1]."""
    n_vars = A.shape[1]
    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)

    for row, b in zip(A, B):
        dim = np.flatnonzero(row)[0]
        bound = b / row[dim]
        if row[dim] > 0:
            upper[dim] = min(upper[dim], bound)
        else:
            lower[dim] = max(lower[dim], bound)

    return lower, upper


def to_point(explanation, n_vars):
    """The point if the explanation fixes all variables, otherwise fail."""
    A, B = parse(explanation, n_vars)
    if is_interval(A):
        lower, upper = interval_bounds(A, B)
        if np.array_equal(lower, upper):
            return lower
    raise Error('The explanation is not a point')


def _number(v):
    # SMT-LIB does not allow the scientific notation of decimals nor negative literals;
    # the shortest representation that is parsed back to the same float
    s = np.format_float_positional(abs(v), trim='0')
    return '(- %s)' % s if v < 0 else s


def from_point(point):
    """The explanation that fixes all variables to the point."""
    return '(and %s)' % ' '.join('(= x%d %s)' % (n + 1, _number(v)) for n, v in enumerate(point))


def _linear(row):
    terms = ['x%d' % (j + 1) if row[j] == 1 else '(* %s x%d)' % (_number(row[j]), j + 1) for j in np.flatnonzero(row)]
    return terms[0] if len(terms) == 1 else '(+ %s)' % ' '.join(terms)


def segment_constraints(p, q):
    """The constraints A x <= B of the segment between the points.

    The features that differ are expressed linearly in the first of them (the reference),
    which is bounded by the endpoints; the other features are fixed.
    """
    n_vars = p.size
    varying = np.flatnonzero(p != q)
    A, B = fix(np.empty((0, n_vars)), np.empty(0), {i: p[i] for i in range(n_vars) if p[i] == q[i]})
    if varying.size == 0:
        return A, B

    k = varying[0]
    rows, rhs = [A], [B]
    for j in varying[1:]:
        # x_j = p_j + c (x_k - p_k)
        c = (q[j] - p[j]) / (q[k] - p[k])
        e = np.zeros(n_vars)
        e[j], e[k] = 1, -c
        rows += [e[None, :], -e[None, :]]
        rhs += [[p[j] - c * p[k]], [c * p[k] - p[j]]]
    e = np.zeros(n_vars)
    e[k] = 1
    rows += [e[None, :], -e[None, :]]
    rhs += [[max(p[k], q[k])], [-min(p[k], q[k])]]
    return np.vstack(rows), np.concatenate(rhs)


def from_segment(p, q):
    """The explanation of the segment between the points."""
    return from_constraints(*segment_constraints(p, q))


def from_constraints(A, B):
    """The explanation of the constraints A x <= B, with the pairs of opposite constraints as equalities."""
    pairs = equalities(A, B)
    seconds = set(pairs.values())
    atoms = []
    for i, (row, b) in enumerate(zip(A, B)):
        if i in seconds:
            continue
        atoms.append('(%s %s %s)' % ('=' if i in pairs else '<=', _linear(row), _number(b)))
    return '(and %s)' % ' '.join(atoms)
