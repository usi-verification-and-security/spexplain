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


class NonConvexError(Error):
    """The explanation is not convex: not a conjunction of linear constraints over the inputs,
    or with a neuron variable that it does not fix."""


class MissingModelError(NonConvexError):
    """The explanation has neuron variables that it does not define: without the model, they are their ReLUs,
    so it is not convex."""


_NEURON_VAR = re.compile(r'l(\d+)_n(\d+)')
_ATOMS = ('<=', '>=', '=')
_NUMBER = re.compile(r'[-+]?[0-9.]+')


# The connectives that make an explanation not convex (in general)
_NON_CONVEX = {'or': 'a disjunction (or)', 'ite': 'an if-then-else (ite), i.e. a case split', '=>':
               'an implication (=>)', 'xor': 'a xor', 'distinct': 'a disequality (distinct)'}


def _check_convex(tree):
    """Fail unless the parsed explanation is a conjunction of (negated) linear atoms."""
    if not isinstance(tree, list) or not tree:
        raise Error(f'Invalid explanation: not a constraint: {tree}')
    op = tree[0]
    if op == 'and':
        for arg in tree[1:]:
            _check_convex(arg)
    elif op in _ATOMS:
        pass
    elif op == 'not' and len(tree) == 2 and isinstance(tree[1], list) and tree[1]:
        arg = tree[1]
        if arg[0] in ('<=', '>='):
            pass
        elif arg[0] in ('and', 'not') and len(arg) == 2:
            _check_convex(['not', arg[1]] if arg[0] == 'and' else arg[1])
        elif arg[0] == '=':
            raise NonConvexError('The explanation is not convex: it has a disequality (negated =)')
        elif arg[0] == 'and':
            raise NonConvexError('The explanation is not convex: it has a negated conjunction, i.e. a disjunction')
        elif arg[0] in _NON_CONVEX or arg[0] == 'not':
            raise NonConvexError(f'The explanation is not convex: it has a negated {arg[0]}')
        else:
            raise Error(f'Unsupported explanation: it has a negated {arg[0]}')
    elif op in _NON_CONVEX:
        raise NonConvexError(f'The explanation is not convex: it has {_NON_CONVEX[op]}')
    else:
        raise Error(f'Unsupported explanation: it has the operator {op}')


def _strip_annotations(tree, marked):
    """The tree without the annotations (! <term> <attribute>...); collects the neuron variables of the attributes
    :neuron <var> into marked."""
    if not isinstance(tree, list):
        return tree
    if tree and tree[0] == '!':
        if len(tree) < 2:
            raise Error(f'Invalid annotation: {tree}')
        attrs = tree[2:]
        for key, value in zip(attrs, attrs[1:]):
            if key == ':neuron' and isinstance(value, str):
                marked.add(value)
        return _strip_annotations(tree[1], marked)
    return [_strip_annotations(t, marked) for t in tree]


def _defined_by_equality(tree, acc):
    """The neuron variables v of the conjuncts (= v <term>) or (= <term> v)."""
    if tree[0] == 'and':
        for arg in tree[1:]:
            _defined_by_equality(arg, acc)
    elif tree[0] == '=':
        acc.update(v for v in tree[1:] if isinstance(v, str) and neuron_var(v))
    return acc


def _eliminate(A, B, columns, names):
    """Eliminate the variables of the columns by the equalities (pairs of opposite rows) of A z <= B."""
    for col in columns:
        pairs = [(i, j) for i, j in equalities(A, B).items() if A[i, col] != 0]
        if not pairs:
            raise Error(f'The neuron variable {names[col]} is defined only by inequalities: not supported yet')
        i, j = max(pairs, key=lambda ij: abs(A[ij[0], col]))
        row, b = A[i].copy(), B[i]
        factors = A[:, col] / row[col]
        A -= np.outer(factors, row)
        B -= factors * b
        A[:, col] = 0
    # The constraints without variables left
    constant = ~A.any(axis=1)
    if np.any(B[constant] < -1e-9 * np.maximum(1, np.abs(B[constant]))):
        raise Error('The explanation is empty after the elimination of the neuron variables')
    return A[~constant], B[~constant]


def _symbols(tree, acc):
    if isinstance(tree, list):
        for t in tree[1:]:
            _symbols(t, acc)
    elif not _NUMBER.fullmatch(tree):
        acc.add(tree)
    return acc


def neuron_var(var):
    """(layer, node from 0) of a neuron variable l<layer>_n<node from 1> (as in the engine), otherwise None."""
    m = _NEURON_VAR.fullmatch(var)
    return (int(m[1]), int(m[2]) - 1) if m else None


def neuron_var_name(layer, node):
    return f'l{layer}_n{node + 1}'


def parse(explanation, n_vars, sample=None, network=None):
    """Parse the explanation into constraints A x <= B.

    It has to be convex (NonConvexError). Its neuron variables l<k>_n<j>, if any:
    - those that it defines, by an equality (= l<k>_n<j> <term>) or by the constraints annotated with
      (! <constraint> :neuron l<k>_n<j>), are eliminated by their equalities (only by inequalities is not supported
      yet), without the model;
    - the others are the ReLUs of the model: without the network (neurons.Network) and the sample, they make the
      explanation not convex (MissingModelError); with them, it is convex if it implies their activations in the
      sample (up to the rounding), and they are substituted (see neurons.py).
    """
    tree = parse_smt2.parse(explanation)
    if len(tree) != 1:
        raise Error(f'Invalid explanation: {explanation[:100]}')
    marked = set()
    tree = _strip_annotations(tree[0], marked)
    _check_convex(tree)

    variables = make_variables(n_vars)
    input_vars = set(variables)
    neurons = set()
    for var in _symbols(tree, set()):
        if var in input_vars:
            continue
        kj = neuron_var(var)
        if kj is None:
            raise Error(f'Unknown variable in the explanation: {var}')
        neurons.add(kj)
    neurons = sorted(neurons)
    defined = {neuron_var(v) for v in marked | _defined_by_equality(tree, set()) if neuron_var(v)}
    undefined = [kj for kj in neurons if kj not in defined]
    if undefined and (network is None or sample is None):
        names = ' '.join(neuron_var_name(*kj) for kj in undefined[:5]) + (' ...' if len(undefined) > 5 else '')
        raise MissingModelError(f'The explanation is not convex: it does not define the neuron variables {names}, '
                                'which are thus their ReLUs (with the model, --model, it can be convex if it '
                                'implies their activations)')
    variables += [neuron_var_name(*kj) for kj in neurons]

    try:
        parsed_ineq = parse_smt2.interpret([tree], variables)
    except parse_smt2.NonLinearError as e:
        raise NonConvexError(f'The explanation is not convex: it has {e}')
    except ValueError as e:
        raise Error(f'Unsupported explanation: it has {e}')
    A, B = -parsed_ineq[:, :-1], parsed_ineq[:, -1]
    if neurons:
        column = {kj: n_vars + i for i, kj in enumerate(neurons)}
        A, B = _eliminate(A.astype(float), B.astype(float), [column[kj] for kj in neurons if kj in defined],
                          variables)
        A = A[:, list(range(n_vars)) + [column[kj] for kj in undefined]]
    if undefined:
        # Only then needs scipy
        from . import neurons as neurons_
        A, B = neurons_.substitute(A, B, n_vars, undefined, network, sample)
    return A, B


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


def to_point(explanation, n_vars, sample=None, network=None):
    """The point if the explanation fixes all variables, otherwise fail."""
    A, B = parse(explanation, n_vars, sample, network)
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
