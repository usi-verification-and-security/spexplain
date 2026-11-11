# The neuron variables l<k>_n<j> of explanations (the activation of the neuron j of the hidden layer k, from 1,
# as in the engine): an explanation that fixes all the neurons that it depends on (implies their activation
# conditions in the sample, as the engine adds them for the fixed neurons) is convex, and its neuron variables are
# the linear functions of the inputs of the activation pattern of the sample; they are substituted numerically,
# layer by layer (without the blow-up of inlining the terms); only of the neuron variables that the explanation does
# not define (see formula.parse)

import numpy as np
import scipy.optimize

from .. import Error
from .formula import NonConvexError, neuron_var_name

# Of the implication of the activation conditions, relative to the weights of the neuron (the coefficients of the
# explanations are rationals with about 6-7 significant digits, so their rounding errors add up)
IMPLICATION_TOLERANCE = 1e-5


class Network:
    """The hidden layers of the model (spexplain.model.Model), with ReLU activations."""

    def __init__(self, model):
        self.weights = model.weights[:-1]
        self.biases = model.biases[:-1]

    def inputs(self, sample):
        """The inputs of the hidden neurons in the sample, per layer."""
        values = np.asarray(sample, dtype=float)
        inputs = []
        for w, b in zip(self.weights, self.biases):
            pre = w @ values + b
            inputs.append(pre)
            values = np.maximum(pre, 0)
        return inputs


def _implies(A, B, n_inputs, objective, const, active, tolerance):
    """Whether A z <= B, with the inputs within [0,1] and the neuron variables nonnegative, implies the activation
    condition of the input objective x + const: >= 0 if active, otherwise <= 0 (up to the tolerance)."""
    n_vars = A.shape[1]
    c = np.zeros(n_vars)
    c[:n_inputs] = objective if active else -objective
    bounds = [(0, 1)] * n_inputs + [(0, None)] * (n_vars - n_inputs)
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=B, bounds=bounds, method='highs')
    if res.status == 2:
        raise Error('The explanation is empty')
    if res.status == 3:
        return False
    if res.status != 0:
        raise Error(f'Linear programming failed: {res.message}')
    # The minimum of the input if active, otherwise its maximum
    value = (res.fun if active else -res.fun) + const
    return value >= -tolerance if active else value <= tolerance


def substitute(A, B, n_inputs, neurons, network, sample):
    """Substitute the neuron variables in the constraints A z <= B over z = (x, the neurons in this order).

    Layer by layer: the explanation has to imply the activation condition in the sample of each neuron of the layer
    that it depends on (all of the layers before the deepest one, and the neurons of the deepest one that occur),
    given the previous layers substituted; otherwise it is not convex (NonConvexError).
    Returns the constraints A x <= B over the inputs.
    """
    depth = max(layer for layer, _ in neurons)
    if depth > len(network.weights):
        raise Error(f'Neuron variables of the layer {depth}, but the model has {len(network.weights)} hidden layers')
    A, B = A.astype(float), B.astype(float)
    column = {kj: n_inputs + i for i, kj in enumerate(neurons)}
    inputs = network.inputs(sample)

    # The values of the neurons of the previous layer as the affine maps M x + c (the inputs at first)
    M = np.eye(n_inputs)
    c = np.zeros(n_inputs)
    for layer in range(1, depth + 1):
        w, b = network.weights[layer - 1], network.biases[layer - 1]
        # The inputs of the neurons of the layer
        P = w @ M
        p = w @ c + b
        active = inputs[layer - 1] >= 0

        nodes = range(len(b)) if layer < depth else [j for k, j in neurons if k == layer]
        for j in nodes:
            tolerance = IMPLICATION_TOLERANCE * max(1, np.abs(w[j]).sum())
            if not _implies(A, B, n_inputs, P[j], p[j], active[j], tolerance):
                raise NonConvexError(f'The explanation is not convex: it does not fix the neuron '
                                     f'{neuron_var_name(layer, j)} (its activation in the sample)')

        M = P * active[:, None]
        c = p * active
        for (k, j), col in column.items():
            if k != layer:
                continue
            A[:, :n_inputs] += np.outer(A[:, col], M[j])
            B -= A[:, col] * c[j]
            A[:, col] = 0

    A = A[:, :n_inputs]
    # The constraints without variables left
    constant = ~A.any(axis=1)
    if np.any(B[constant] < 0):
        raise Error('The explanation is empty after the substitution of the neurons')
    return A[~constant], B[~constant]
