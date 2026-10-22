# Domains of the features in real units, given by a CSV file with the columns Min, Max and optionally Var, Feature
# (e.g. 'x1,Age,29,77'): without Var, the rows are the features in their order;
# without Feature, the names of the features are those of the dataset;
# the features that are not listed keep the domain [0,1]
# Only used for the outputs of visualizations, the inputs are still assumed to be normalized to [0,1]

import csv
import re

import numpy as np

from . import Error


class Domains:
    def __init__(self, path, dataset):
        n_vars = dataset.n_features()
        self.lower = np.zeros(n_vars)
        self.upper = np.ones(n_vars)
        self.names = list(dataset.feature_names)

        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            missing = {'Min', 'Max'} - set(reader.fieldnames or [])
            if missing:
                raise Error(f'Missing the columns {", ".join(sorted(missing))} in the domains: {path}')
            for k, row in enumerate(reader):
                var = (row.get('Var') or '').strip()
                if var:
                    # Also 'X10'
                    m = re.fullmatch(r'x(\d+)', var.lower())
                    i = int(m[1]) - 1 if m else -1
                else:
                    i = k
                if not 0 <= i < n_vars:
                    raise Error(f'Invalid variable for {n_vars} features in the domains: {var or f"row {k + 1}"}')
                try:
                    self.lower[i], self.upper[i] = float(row['Min']), float(row['Max'])
                except ValueError:
                    raise Error(f'Invalid domain of x{i + 1}: {row["Min"]}, {row["Max"]}')
                if self.lower[i] >= self.upper[i]:
                    raise Error(f'Empty domain of x{i + 1}: {row["Min"]}, {row["Max"]}')
                name = (row.get('Feature') or '').strip()
                if name:
                    self.names[i] = name

    def range(self):
        return self.upper - self.lower

    def scale(self, x, i=None):
        """The normalized values of all features (or of the feature i) in the real units."""
        if i is None:
            return self.lower + x * self.range()
        return self.lower[i] + x * (self.upper[i] - self.lower[i])

    def scale_constraints(self, A, B):
        """The constraints A x <= B over the normalized features as constraints over the features in the real units."""
        A = A / self.range()
        return A, B + A @ self.lower
