# Neural network classifiers in the NNet format, evaluated as in the engine:
# ReLU hidden layers, a linear output layer, the normalization information is ignored;
# a single output means a binary classification (class 1 iff the output is >= 0), otherwise the maximal output

import numpy as np

from nnet_utils.readNNet import readNNet

from . import Error


class Model:
    def __init__(self, path):
        try:
            weights, biases = readNNet(str(path))
        except (ValueError, IndexError):
            raise Error(f'Invalid model in the NNet format: {path}')
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]

    def n_inputs(self):
        return self.weights[0].shape[1]

    def n_classes(self):
        n_outputs = self.weights[-1].shape[0]
        return 2 if n_outputs == 1 else n_outputs

    def outputs(self, X):
        """The output values of the samples given by the rows of X."""
        values = np.asarray(X, dtype=float)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            values = values @ w.T + b
            if k < len(self.weights) - 1:
                values = np.maximum(values, 0)
        return values

    def classify(self, X):
        """The classes of the samples given by the rows of X."""
        values = self.outputs(X)
        if values.shape[1] == 1:
            return (values[:, 0] >= 0).astype(int)
        return np.argmax(values, axis=1)
