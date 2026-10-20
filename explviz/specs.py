"""Concrete DatasetSpec instances for the datasets in this repo."""

from __future__ import annotations

from .config import DatasetSpec

#: Human-readable names for the 13 heart-attack features, in x1..x13 order.
HEART_ATTACK_LABELS = [
    "Age",
    "Sex",
    "Chest pain",
    "Blood pressure",
    "Cholesterol [mg/dl]",
    "Fasting blood sugar",
    "Resting electrocardiographic",
    "Maximum heart rate",
    "Exercise induced angina",
    "Oldpeak",
    "Peak slope",
    "Major vessels",
    "Thal",
]

#: Raw (unscaled) feature ranges, as used by ``heartAttack.csv``.
HEART_ATTACK_LOWER = [29.0, 0.0, 0.0, 94.0, 126.0, 0.0, 0.0, 71.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HEART_ATTACK_UPPER = [77.0, 1.0, 3.0, 200.0, 564.0, 1.0, 2.0, 202.0, 1.0, 6.2, 2.0, 4.0, 3.0]

_VARS = [f"x{i}" for i in range(1, 14)]

#: For the older figures that read the raw ``heartAttack.csv`` (303 rows).
HEART_ATTACK_RAW = DatasetSpec(
    name="heart-attack-raw",
    var_names=_VARS,
    axis_labels=HEART_ATTACK_LABELS,
    bounds=list(zip(HEART_ATTACK_LOWER, HEART_ATTACK_UPPER)),
    scaled=False,
)

#: For ``datasets/heart_attack_s100_scaled.csv`` (100 rows, min-max scaled to [0,1]).
HEART_ATTACK_SCALED = DatasetSpec(
    name="heart-attack-scaled",
    var_names=_VARS,
    axis_labels=HEART_ATTACK_LABELS,
    bounds=[(0.0, 1.0)] * 13,
    scaled=True,
)


def mnist_spec(n_pixels: int = 784) -> DatasetSpec:
    """Spec for the MNIST explanation files (same grammar, x1..x784)."""
    return DatasetSpec(
        name=f"mnist-{n_pixels}",
        var_names=[f"x{i}" for i in range(1, n_pixels + 1)],
        axis_labels=[f"px{i}" for i in range(1, n_pixels + 1)],
        bounds=[(0.0, 1.0)] * n_pixels,
        scaled=True,
    )
