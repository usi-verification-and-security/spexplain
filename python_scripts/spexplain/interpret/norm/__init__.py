# Technique norm: the point of an explanation with the minimal (L2) norm, the inverted minimal norm, or the maximal norm
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


# obtain points with minimum norm in the inverted domains, i.e., closest to their upper corner
def get_inv_norm_point(A, B, sample):
    # x = 1 - y maps the domains [0,1] onto themselves and A x <= B to -A y <= B - A 1
    ones = np.ones(A.shape[1])
    return ones - get_min_norm_point(-A, B - A @ ones, ones - sample)


def check_point(A, B, point):
    """Check that the point satisfies the explanation and the domains."""
    assert np.all(np.dot(A, point) <= B + TOLERANCE)
    assert np.all(point >= 0.0)
    assert np.all(point <= 1.0)


class Config:
    """The configuration of the technique given by its parameters."""

    def __init__(self, params):
        # The kinds of the points in their order: 'min', 'inv' or 'max'
        self.kinds = []
        for param in params:
            if param not in ('min', 'inv', 'max'):
                raise Error('Unknown parameter of norm: %s' % param)
            if param in self.kinds:
                raise Error('Duplicate parameter of norm: %s' % param)
            self.kinds.append(param)
        if not self.kinds:
            self.kinds = ['min']


def interpret(explanation, config, sample):
    """Yields the points of the explanation of the sample with the minimal, the inverted minimal, or the maximal
    norm, one for each kind, each as soon as it is computed."""
    A, B = formula.parse(explanation, sample.size, sample, getattr(config, 'network', None))

    for kind in config.kinds:
        if formula.is_interval(A):
            # The domains are non-negative, so the upper bounds are both the closest to the upper corner and maximal
            lower, upper = formula.interval_bounds(A, B)
            point = lower if kind == 'min' else upper
        elif kind == 'min':
            point = get_min_norm_point(A, B, sample)
        elif kind == 'inv':
            point = get_inv_norm_point(A, B, sample)
        else:
            point = get_max_norm_point(A, B, sample)

        check_point(A, B, point)
        yield point
