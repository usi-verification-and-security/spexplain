#!/usr/bin/env python3
## The relaxed features of explanations: a feature is relaxed if the explanation (within the domains [0,1])
## contains points that differ in it, i.e. it is not fixed to the value of the sample;
## and the widths of each feature, as fractions of [0,1]: the inner one, of the axis-parallel segment through the
## sample within the explanation (only the feature changes, the others stay at the sample), and the outer one, of
## the projection of the explanation (the others may change too; the projections form its bounding box). The inner
## width is at most the outer one, and they coincide for the interval explanations (e.g. of VeriX-plus and CORA);
## the means are over all the features (the fixed ones are 0). And the Linf diameter: of the largest Linf ball of the
## sample within the explanation (its robustness), i.e. the cube of the sample +-t in all the features, within
## [0,1], as the mean width of its features (as above); it is 0 if any feature is fixed (e.g. by most of the
## explanations of VeriX-plus and CORA, otherwise 2 epsilon up to [0,1]). Per feature, diameter <= inner <= outer.
##
## Only of the convex explanations, i.e. the conjunctions of linear constraints over the inputs: the others
## (disjunctions, ite, ...) are only counted. The neuron variables l<k>_n<j> (the activation of the neuron j of
## the hidden layer k) that the explanation defines (by equalities) are eliminated; the others need the model: the
## explanation is convex if it implies the activation condition (in the sample) of each neuron that it depends on (as for the neurons fixed by the engine), and then the neurons are
## substituted as the linear functions of the inputs that they are; otherwise it is not convex (e.g. with unfixed
## neurons). The explanations are parsed as by spexplain interpret (formula.parse, see interpret/neurons.py).
##
## For the interval explanations (e.g. of VeriX-plus and CORA), directly from the bounds; for the others, by linear
## programming (scipy, HiGHS): one LP for whether the explanation has an interior (relative to its equalities),
## then all features but those fixed by the equalities are relaxed, otherwise, and for the outer widths, two LPs
## per feature. The explanations are parsed into floating-point numbers: a feature whose outer
## width is below TOLERANCE counts as fixed, while the syntactic equalities are recognized exactly. The inner widths
## directly from the slacks of the constraints in the sample, the Linf diameter by bisection over t (the maximum
## of a constraint over a box is per feature at its bound, and it grows with t), without LPs.
##
## USAGE: relaxed_features.py <explanations> <dataset> [--model <nnet>] [--outer]
##   <explanations>  file with the explanations in the SMT-LIB2 format, one per sample (e.g. phi.txt)
##   <dataset>       the CSV file of the samples
##   --model <nnet>  the classifier in the NNet format, for the explanations with neuron variables
##   --outer         also compute the outer widths of the features of the non-interval explanations (by LPs; of
##                   the interval ones always)
## Prints a line per explanation: <relaxed features> <features> <Linf diameter> <mean inner width>
## <mean outer width> <kind>,
## the kind is convex, nonconvex, nomodel (with neuron variables that it does not define, but without the model),
## or timeout ('<null>');
## the numbers are '-' if not computed

import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'python_scripts'))
from spexplain.dataset import Dataset  # noqa: E402
from spexplain.interpret import formula  # noqa: E402
from spexplain.interpret.formula import MissingModelError, NonConvexError  # noqa: E402

TOLERANCE = 1e-6
# Of the sample satisfying the explanation
SAMPLE_TOLERANCE = 1e-6
BISECTION_ITERATIONS = 50


def _solve(c, A_ub, B_ub, A_eq, B_eq, bounds):
    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=B_ub, A_eq=A_eq if len(A_eq) else None,
                                 b_eq=B_eq if len(B_eq) else None, bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(f'linear programming failed: {res.message}')
    return res


class Problem:
    """The explanation as A_ub x <= B_ub, A_eq x = B_eq, the inputs within [0,1]."""

    def __init__(self, A, B):
        # The equalities (pairs of opposite rows) separately
        pairs = formula.equalities(A, B)
        eq_rows = sorted(pairs)
        is_eq = np.zeros(len(A), dtype=bool)
        is_eq[eq_rows] = True
        is_eq[list(pairs.values())] = True
        self.A_ub, self.B_ub = A[~is_eq], B[~is_eq]
        self.A_eq, self.B_eq = A[eq_rows], B[eq_rows]
        self.n_vars = A.shape[1]

    def contains(self, x):
        """Whether the point satisfies the constraints, up to SAMPLE_TOLERANCE (relative)."""
        for A, B, eq in ((self.A_ub, self.B_ub, False), (self.A_eq, self.B_eq, True)):
            if not len(A):
                continue
            scale = np.maximum(1, np.abs(A).max(axis=1))
            diff = A @ x - B
            if np.any((np.abs(diff) if eq else diff) > SAMPLE_TOLERANCE * scale):
                return False
        return True

    def outer_widths(self, features):
        """The outer widths (of the projections) of the given features (the others are 0), by two LPs per feature."""
        widths = np.zeros(self.n_vars)
        for i in features:
            c = np.zeros(self.n_vars)
            c[i] = 1
            lower = _solve(c, self.A_ub, self.B_ub, self.A_eq, self.B_eq, (0, 1)).fun
            upper = -_solve(-c, self.A_ub, self.B_ub, self.A_eq, self.B_eq, (0, 1)).fun
            widths[i] = max(upper - lower, 0)
        return widths

    def has_interior(self):
        """Whether the inequalities (with the domains) have an interior relative to the equalities:
        the largest ball (Chebyshev) within them has a positive radius."""
        n = self.n_vars
        eye = np.eye(n)
        A = np.vstack([self.A_ub, eye, -eye])
        B = np.concatenate([self.B_ub, np.ones(n), np.zeros(n)])
        norms = np.linalg.norm(A, axis=1)
        # Variables x, r: maximize r subject to A x + |A_i| r <= B
        c = np.zeros(n + 1)
        c[-1] = -1
        A_eq_r = np.hstack([self.A_eq, np.zeros((len(self.A_eq), 1))]) if len(self.A_eq) else self.A_eq
        res = _solve(c, np.hstack([A, norms[:, None]]), B, A_eq_r, self.B_eq, [(None, None)] * n + [(0, 1)])
        return -res.fun > TOLERANCE

    def fixed_by_equalities(self):
        """The features that the equalities determine: those zero in their null space."""
        if not len(self.A_eq):
            return np.zeros(self.n_vars, dtype=bool)
        null = scipy.linalg.null_space(self.A_eq)
        return np.abs(null).max(axis=1, initial=0) <= TOLERANCE


def inner_widths(A, B, sample):
    """The widths of the axis-parallel segments through the sample within A x <= B and [0,1], per feature:
    from the slacks of the constraints in the sample (within SAMPLE_TOLERANCE, relative)."""
    slack = np.maximum(B - A @ sample, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        steps = np.where(A != 0, slack[:, None] / np.abs(A), np.inf)
    up = np.where(A > 0, steps, np.inf).min(axis=0, initial=np.inf)
    down = np.where(A < 0, steps, np.inf).min(axis=0, initial=np.inf)
    return np.minimum(sample + up, 1) - np.maximum(sample - down, 0)


def linf_diameter(A, B, sample):
    """The mean width of the features of the largest Linf ball of the sample +-t within [0,1] that satisfies
    A x <= B, up to SAMPLE_TOLERANCE (relative), with t by bisection (up to 1; below TOLERANCE as 0)."""
    tolerance = SAMPLE_TOLERANCE * np.maximum(1, np.abs(A).max(axis=1))

    def box(t):
        return np.maximum(sample - t, 0), np.minimum(sample + t, 1)

    def within(t):
        lower, upper = box(t)
        # The maximum of each constraint over the box
        return np.all(np.where(A > 0, A * upper, A * lower).sum(axis=1) <= B + tolerance)

    low, high = 0.0, 1.0
    if within(high):
        low = high
    for _ in range(BISECTION_ITERATIONS if low < high else 0):
        mid = (low + high) / 2
        low, high = (mid, high) if within(mid) else (low, mid)
    if low <= TOLERANCE:
        return 0.0
    lower, upper = box(low)
    return (upper - lower).mean()


def analyze(explanation, sample, network, with_outer):
    """-> (kind, the number of the relaxed features, the Linf diameter, the mean inner width, the mean outer width
    (None if not computed)), the numbers None if not convex"""
    n_vars = len(sample)
    if np.any(sample < 0) or np.any(sample > 1):
        raise RuntimeError('the sample is outside the domains [0,1]')
    try:
        A, B = formula.parse(explanation, n_vars, sample, network)
    # A missing model is also not convex (the undefined neuron variables are their ReLUs), only reported apart
    except MissingModelError:
        return 'nomodel', None, None, None, None
    except NonConvexError:
        return 'nonconvex', None, None, None, None

    # The interval explanations (also with the equalities of the fixed features) directly
    if formula.is_interval(A):
        lower, upper = formula.interval_bounds(A, B)
        widths = np.maximum(upper - lower, 0)
        relaxed = widths > TOLERANCE
        return ('convex', int(np.sum(relaxed)), linf_diameter(A, B, sample), inner_widths(A, B, sample).mean(),
                widths.mean())

    problem = Problem(A, B)
    if not problem.contains(sample):
        raise RuntimeError('the sample does not satisfy the explanation')
    if problem.has_interior():
        relaxed = np.flatnonzero(~problem.fixed_by_equalities())
        outer = problem.outer_widths(relaxed).mean() if with_outer else None
        return 'convex', len(relaxed), linf_diameter(A, B, sample), inner_widths(A, B, sample).mean(), outer

    widths = problem.outer_widths(range(n_vars))
    return ('convex', int(np.sum(widths > TOLERANCE)), linf_diameter(A, B, sample),
            inner_widths(A, B, sample).mean(), widths.mean())


def main(args):
    with_outer = '--outer' in args
    args = [a for a in args if a != '--outer']
    model = None
    if '--model' in args:
        i = args.index('--model')
        if i + 1 >= len(args):
            print('Missing the model after --model', file=sys.stderr)
            return 1
        model = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        print('USAGE: relaxed_features.py <explanations> <dataset> [--model <nnet>] [--outer]', file=sys.stderr)
        return 1
    dataset = Dataset(args[1])
    n_vars = dataset.n_features()
    network = None
    if model:
        from spexplain.model import Model
        from spexplain.interpret.neurons import Network
        network = Network(Model(model))
        if network.weights[0].shape[1] != n_vars:
            print(f'The number of the inputs of the model ({network.weights[0].shape[1]}) does not match the number '
                  f'of the features of the dataset ({n_vars})', file=sys.stderr)
            return 1

    idx = 0
    with open(args[0]) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            idx += 1
            if line == '<null>':
                print('- - - - - timeout')
                continue
            if idx > len(dataset):
                print(f'{args[0]}: more explanations than samples in {args[1]}', file=sys.stderr)
                return 1
            try:
                kind, relaxed, diameter, inner, outer = analyze(line, dataset.samples[idx - 1], network, with_outer)
            except Exception as e:
                print(f'{args[0]}: explanation {i}: {e}', file=sys.stderr)
                return 1
            print('-' if relaxed is None else relaxed, n_vars,
                  *('-' if v is None else f'{v:.6g}' for v in (diameter, inner, outer)), kind, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
