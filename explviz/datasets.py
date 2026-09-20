"""Loading sample points and explanation formula files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_points(csv_path: str | Path, has_label: bool = True):
    """Load a dataset CSV.

    Returns ``(X, y)`` where ``X`` is (n_samples, n_features) float and ``y``
    is the label column (or ``None``).  Row ``i`` of ``X`` corresponds to line
    ``i`` of every ``.phi.txt`` file for that dataset.
    """
    df = pd.read_csv(csv_path)
    if has_label:
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy()
    else:
        X = df.to_numpy(dtype=float)
        y = None
    return X, y


def load_explanations(path: str | Path) -> list[str]:
    """Read a ``.phi.txt``: one formula per non-blank line."""
    text = Path(path).read_text()
    return [line for line in text.splitlines() if line.strip()]


def select_explanation(
    formulas: Sequence[str],
    sample_index: int,
    index_map: dict | None = None,
) -> str:
    """Pick the formula for ``sample_index``.

    Some sources (notably VeriX) only contain a subset of samples, in their own
    order; ``index_map`` maps a dataset sample index to a line number in that
    file.  This replaces the bare module-level ``verix_transform`` dict.
    """
    line = sample_index if index_map is None else index_map[sample_index]
    if not 0 <= line < len(formulas):
        raise IndexError(
            f"sample {sample_index} maps to line {line}, "
            f"but the file has {len(formulas)} formulas"
        )
    return formulas[line]


def sample_point(X: np.ndarray, i: int) -> np.ndarray:
    """Feature vector of sample ``i`` as a plain float array."""
    return np.asarray(X[i], dtype=float).ravel()
