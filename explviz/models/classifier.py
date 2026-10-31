"""The single copy of the feed-forward model wrapper.

``SimpleNN`` + ``read_model`` were duplicated verbatim in at least eight
scripts.  This version adds batched prediction, so the class-region background
costs one forward pass instead of one per sampled point.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .nnet import read_nnet


class SimpleNN(nn.Module):
    """Fully connected ReLU network; the final layer is left as a raw logit."""

    def __init__(self, weights, biases):
        super().__init__()
        self.layers = nn.ModuleList()
        for w, b in zip(weights, biases):
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32))
            layer.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32))
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class Classifier:
    """Binary classifier over the input space, with batched inference."""

    def __init__(self, module: nn.Module):
        self.module = module.eval()

    @classmethod
    def from_nnet(cls, path) -> "Classifier":
        weights, biases = read_nnet(path)
        return cls(SimpleNN(weights, biases))

    @torch.no_grad()
    def logits(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        out = self.module(torch.from_numpy(X))
        return out.squeeze(-1).cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        """P(class 1) for each row of ``X``."""
        return 1.0 / (1.0 + np.exp(-self.logits(X)))

    def predict(self, X) -> np.ndarray:
        """Predicted class (0/1) for each row of ``X``."""
        return (self.predict_proba(X) >= 0.5).astype(int)
