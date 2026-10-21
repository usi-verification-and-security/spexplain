# Explanations as formulas in the SMT-LIB format, over the variables x1, ..., xN as in the engine

import numpy as np

from .. import Error
from . import parse_smt2


def make_variables(n_vars):
    return ['x%d' % (n + 1) for n in range(n_vars)]


def parse(explanation, n_vars):
    """Parse the explanation into constraints A x <= B."""
    parsed_ineq = parse_smt2.parse_and_interpret(explanation, make_variables(n_vars))
    return -parsed_ineq[:, :-1], parsed_ineq[:, -1]


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


def from_point(point):
    """The explanation that fixes all variables to the point."""
    # SMT-LIB does not allow the scientific notation of decimals nor negative literals;
    # the shortest representation that is parsed back to the same float
    def to_str(v):
        s = np.format_float_positional(abs(v), trim='0')
        return '(- %s)' % s if v < 0 else s

    values = (to_str(v) for v in point)
    return '(and %s)' % ' '.join('(= x%d %s)' % (n + 1, v) for n, v in enumerate(values))
