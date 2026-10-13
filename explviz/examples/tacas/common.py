"""Shared configuration for the TACAS slice figures.

Datasets, networks, methods, where things live, and a parse cache: parsing a
heart-attack formula costs ~0.12 s, and the selection step reads 100 formulas
per (network, method), so the parsed polytopes are pickled next to the output.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from explviz import Classifier, DatasetSpec, load_explanations, load_points, parse_formula
from explviz.parsing.smt2 import FormulaError
from explviz.paths import data_path
from explviz.specs import HEART_ATTACK_LABELS
from explviz.style import DEFAULT_STYLE

#: Repository root (…/explviz/examples/tacas/common.py -> parents[3]).
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "plots" / "tacas"
CACHE_DIR = OUT_DIR / ".cache"

NETWORKS = ["50x1", "50x4"]
#: Method keys as they appear in the explanation directory names.
METHODS = ["fix-all", "guide-all", "naive"]
LABELS = {"naive": "Direct", "fix-all": "Fix-all", "guide-all": "Guide-all"}
ITP = "itp_aweak_bstrong"
#: Interpolation settings compared by itp_compare.py, display name -> file stem.
ITPS = {"strong": "itp_astrong_bstrong", "mid": "itp_afactor_0.5_bstrong",
        "weak": "itp_aweak_bstrong"}

#: Fixed colour/marker per method, the same in every figure (ATVA26 mapping:
#: fix-all red, guide-all green, direct blue).
METHOD_SERIES = {"fix-all": 0, "guide-all": 2, "naive": 3}

LOAN_LABELS = [
    "Age", "Income", "Employment exp.", "Loan amount", "Interest rate",
    "Loan / income", "Credit history length", "Credit score", "Gender: male",
    "Edu: Bachelor", "Edu: Doctorate", "Edu: High school", "Edu: Master",
    "Home: other", "Home: own", "Home: rent", "Intent: education",
    "Intent: home impr.", "Intent: medical", "Intent: personal",
    "Intent: venture", "Previous default",
]


@dataclass(frozen=True)
class Dataset:
    name: str
    spec: DatasetSpec
    class_labels: tuple  # (class 0, class 1)

    @property
    def csv(self) -> Path:
        return data_path(f"datasets/{self.name}/{self.name}_s100_scaled.csv")

    def model(self, net: str) -> Path:
        return data_path(f"models/{self.name}/{self.name}_{net}.nnet")

    def phi(self, net: str, method: str, itp: str = ITP) -> Path:
        return data_path(f"explanations/{self.name}/{net}/s100_scaled/{method}/{itp}.phi.txt")


def _scaled_spec(name, labels):
    n = len(labels)
    return DatasetSpec(name=name, var_names=[f"x{i}" for i in range(1, n + 1)],
                       axis_labels=list(labels), bounds=[(0.0, 1.0)] * n, scaled=True)


DATASETS = {
    # class 0 / class 1 naming follows the existing explviz heart-attack style
    "heart_attack": Dataset("heart_attack", _scaled_spec("heart-attack-scaled", HEART_ATTACK_LABELS),
                            tuple(DEFAULT_STYLE.class_labels)),
    # loan_status: 1 = approved (every applicant with a previous default is 0)
    "loan": Dataset("loan", _scaled_spec("loan-scaled", LOAN_LABELS), ("Rejected", "Approved")),
}


def slug(text: str) -> str:
    """Filename-safe feature name: 'Loan / income' -> 'loan-income'."""
    out = "".join(c.lower() if c.isalnum() else "-" for c in text)
    return "-".join(p for p in out.split("-") if p)


def load_points_for(ds: Dataset):
    return load_points(ds.csv)


def load_classifier(ds: Dataset, net: str) -> Classifier:
    return Classifier.from_nnet(ds.model(net))


def variant_dir(ds: Dataset) -> Path:
    """Output folder of the method (variant) comparison, under its current name."""
    for name in ("variant_compare", "variants_compare"):
        if (OUT_DIR / ds.name / name).is_dir():
            return OUT_DIR / ds.name / name
    return OUT_DIR / ds.name / "variant_compare"


def itp_dir(ds: Dataset) -> Path:
    """Output folder of the interpolation-strength comparison."""
    return OUT_DIR / ds.name / "itp_compare"


def load_formulas(ds: Dataset, net: str, method: str, itp: str = ITP) -> list:
    """Raw formula per sample; ``None`` where the run produced ``<null>``."""
    path = ds.phi(net, method, itp)
    if not path.exists():
        return [None] * 100
    return [f if f.lstrip().startswith("(") else None for f in load_explanations(path)]


def load_polytopes(ds: Dataset, net: str, method: str, itp: str = ITP) -> list:
    """Parsed polytope list per sample (``None`` for missing explanations), cached."""
    path = ds.phi(net, method, itp)
    if not path.exists():
        return [None] * 100
    digest = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    cache = CACHE_DIR / f"{ds.name}_{net}_{method}_{itp}_{digest}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    var_names = ds.spec.var_names
    polys = [None if f is None else _parse_or_none(f, var_names)
             for f in load_formulas(ds, net, method, itp)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(polys))
    return polys


def _parse_or_none(formula, var_names):
    """Parsed polytopes, or ``None`` for formulas explviz cannot draw.

    Some 50x4 fix-all formulas contain ``ite`` inside linear terms (a union
    of polytopes, not one), and an older loan run refers to hidden neurons
    (``l1_n1``); the parser rejects both.
    """
    try:
        return parse_formula(formula, var_names)
    except (FormulaError, ValueError):
        return None


def axis_candidates(X: np.ndarray, min_levels: int = 3) -> list:
    """Features worth putting on an axis: binary / one-hot features are not.

    A binary feature only takes the values 0 and 1, so the slice between them
    is model behaviour on inputs that never occur.
    """
    return [j for j in range(X.shape[1]) if len(np.unique(X[:, j])) >= min_levels]


def pair_stem(ds: Dataset, sample: int, a: int, b: int) -> str:
    """Model-independent part of a name: ``loan_s042_x5-x8_interest-rate_vs_credit-score``."""
    labels = ds.spec.axis_labels
    return (f"{ds.name}_s{sample:03d}_x{a + 1}-x{b + 1}_"
            f"{slug(labels[a])}_vs_{slug(labels[b])}")


def figure_stem(ds: Dataset, net: str, sample: int, a: int, b: int, rank: int | None = None) -> str:
    """Traceable figure name.

    ``r03_loan-50x1_s042_x5-x8_interest-rate_vs_credit-score``: rank in the
    selection, dataset, network, 0-based sample index (= CSV data row = line
    ``sample+1`` of every .phi.txt), then the x-axis and y-axis features as
    1-based SMT variables (x5 is column 5 of the CSV) and their names.
    """
    labels = ds.spec.axis_labels
    stem = (f"{ds.name}-{net}_s{sample:03d}_x{a + 1}-x{b + 1}_"
            f"{slug(labels[a])}_vs_{slug(labels[b])}")
    return stem if rank is None else f"r{rank:02d}_{stem}"
