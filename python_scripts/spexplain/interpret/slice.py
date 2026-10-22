# Technique slice: the slice of an explanation in the selected features through the sample,
# i.e. the values of the sample are substituted for all other features
# Only 2D slices are supported

from .. import Error
from . import formula

N_VARS = 2


class Config:
    """The configuration of the technique given by its parameters."""

    def __init__(self, params, n_vars):
        self.var_indices = None

        names = set()
        for param in params:
            name, *args = param.split()
            if name in names:
                raise Error(f'Duplicate parameter of slice: {name}')
            names.add(name)
            if name == 'vars':
                self.var_indices = parse_vars(args, n_vars, 'slice')
            else:
                raise Error(f'Unknown parameter of slice: {param}')

        if self.var_indices is None:
            raise Error('Technique slice requires the parameter vars')


def parse_vars(args, n_vars, technique):
    """Exactly the supported number of features."""
    var_indices = formula.parse_vars(args, n_vars, technique)
    if len(var_indices) != N_VARS:
        raise Error(f'Technique {technique} supports only {N_VARS} features, got: {" ".join(args)}')
    return var_indices


def fixed_values(config, sample):
    """The values of the sample of the features outside of the slice."""
    return {i: v for i, v in enumerate(sample) if i not in config.var_indices}


def interpret(explanation, config, sample):
    """The constraints A x <= B of the slice; the features outside of the slice have zero columns."""
    A, B = formula.parse(explanation, sample.size)
    return formula.substitute(A, B, fixed_values(config, sample))
