# Technique norm: the point of an explanation with the minimal or the maximal (L2) norm
# Assumes that the domains of all features are normalized to [0,1]

import numpy as np
import scipy.linalg
import scipy.optimize

from ... import Error
from .. import formula

# Tolerance of checking that a point satisfies the explanation
TOLERANCE = 0.001


def _check_start(A, B, start):
    if np.any(np.dot(A, start) > B):
        raise Error('The sample does not satisfy the explanation')


def _minimize(fun, A, B, start, n_vars):
    bounds = [(0.0, 1.0)] * n_vars
    constraints = [scipy.optimize.LinearConstraint(A, -np.inf, B)]
    res = scipy.optimize.minimize(fun, start, constraints=constraints, bounds=bounds, method='SLSQP',
                                  options={'maxiter': 1000})
    if res.status != 0:
        raise Error('Optimization did not succeed: %s' % res.message)
    return res.x


# obtain points with minimum norm
def get_min_norm_point(A, B, sample):
    _check_start(A, B, sample)
    return _minimize(scipy.linalg.norm, A, B, sample, A.shape[1])


# obtain points with maximum norm
def get_max_norm_point(A, B, sample):
    _check_start(A, B, sample)
    return _minimize(lambda x: 1.0 / scipy.linalg.norm(x), A, B + 0.00001, sample, A.shape[1])


def _reflect_polytope(A, b):
    # reflect at the hyperplane with normal (1,...,1) going through (0.5,...,0.5)
    d = A.shape[1]
    ones = np.ones(d)

    # Reflection matrix
    R = np.eye(d) - (2.0 / d) * np.outer(ones, ones)

    # Reflection translation
    t = (2.0 / d) * ones

    # Reflected polytope
    A_ref = A @ R
    b_ref = b + A @ t

    return A_ref, b_ref


def _reflect_point(x):
    d = x.size
    return x - (2.0 / d) * (np.sum(x) - 1.0)


#+ WIP, not used yet
# obtain points with minimum norm in the inverse space
def get_inv_norm_point(A, B, sample):
    # reflect at the hyperplane with normal (1,...,1) going through (0.5,...,0.5)
    A_ref, B_ref = _reflect_polytope(A, B)

    start = _reflect_point(sample)
    _check_start(A_ref, B_ref, start)

    min_point = _minimize(scipy.linalg.norm, A_ref, B_ref, start, A.shape[1])
    inverted = _reflect_point(min_point)

    assert np.all(np.dot(A_ref, min_point) <= B_ref + TOLERANCE)
    assert np.all(min_point >= 0.0)
    assert np.all(min_point <= 1.0)
    check_point(A, B, inverted)

    return inverted


def check_point(A, B, point):
    """Check that the point satisfies the explanation and the domains."""
    assert np.all(np.dot(A, point) <= B + TOLERANCE)
    assert np.all(point >= 0.0)
    assert np.all(point <= 1.0)


class Config:
    """The configuration of the technique given by its parameters."""

    def __init__(self, params):
        if params in ([], ['min']):
            self.maximal = False
        # Inconsistent: 'inv' yields the point with the maximal norm, not the one from get_inv_norm_point
        elif params == ['inv']:
            self.maximal = True
        else:
            raise Error('Unknown parameters of norm: %s' % ', '.join(params))


def interpret(explanation, config, sample):
    """The point of the explanation of the sample with the minimal or the maximal norm."""
    A, B = formula.parse(explanation, sample.size)

    if formula.is_interval(A):
        # The domains are non-negative
        lower, upper = formula.interval_bounds(A, B)
        point = upper if config.maximal else lower
    elif config.maximal:
        point = get_max_norm_point(A, B, sample)
    else:
        point = get_min_norm_point(A, B, sample)

    check_point(A, B, point)

    return point
