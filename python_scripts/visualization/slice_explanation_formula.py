"""Cut one sample's explanation down to a 2D SMT2 formula on a chosen axis pair.

Every feature except --x and --y is pinned to the sample's values, which is
exactly the slice a snapshot plot draws.  The sliced formula is printed, checked
numerically against the original (both must give the same snapshot polygon),
and optionally saved as a one-line .phi.txt.

Run from the repo root:

    PYTHONPATH=. python3 python_scripts/visualization/slice_explanation_formula.py \
        --index 71 \
        --exp data/explanations/heart_attack/50x1/s100_scaled/fix-all/itp_aweak_bstrong.phi.txt \
        --x x10 --y x1 \
        --out plots/ha_s71_sliced_x10_x1
"""

import argparse
from pathlib import Path

import numpy as np

from explviz import load_explanations, load_points, parse_formula, region_2d, slice_formula
from explviz.paths import data_path
from explviz.specs import HEART_ATTACK_SCALED as spec


def feature_index(s):
    """'x10' -> 9 (1-based variable name); '9' -> 9 (0-based index)."""
    return int(s[1:]) - 1 if s.startswith("x") else int(s)


def phi_path(out):
    """Append '.phi.txt' unless already there; with_suffix would cut 'a0.5' to 'a0'."""
    out = Path(out)
    return out if out.name.endswith(".phi.txt") else out.with_name(out.name + ".phi.txt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", "-i", type=int, required=True,
                   help="sample index (= line index in the .phi.txt)")
    p.add_argument("--exp", required=True, help="explanation .phi.txt file")
    p.add_argument("--x", required=True, help="first free feature: 'x10' or 0-based index '9'")
    p.add_argument("--y", required=True, help="second free feature: 'x1' or 0-based index '0'")
    p.add_argument("--out", "-o",
                   help="output path; '.phi.txt' is appended if missing (default: print only)")
    p.add_argument("--data",
                   default=str(data_path("datasets/heart_attack/heart_attack_s100_scaled.csv",
                                         must_exist=False)),
                   help="dataset CSV the sample index refers to")
    return p.parse_args()


def main():
    args = parse_args()
    X, _ = load_points(args.data)
    i = args.index
    axes = (feature_index(args.x), feature_index(args.y))

    formulas = load_explanations(args.exp)
    if not 0 <= i < len(formulas):
        raise SystemExit(f"{args.exp} has {len(formulas)} formulas; index {i} is out of range")
    original, point = formulas[i], X[i]

    sliced = slice_formula(original, spec.var_names, point, axes)
    print(sliced)
    print(f"\n{len(original)} -> {len(sliced)} characters")

    # Numeric check: both formulas give the same snapshot polygon, disjunct by disjunct.
    regions = [[region_2d(p, axes, spec.bounds, "snapshot", point=point)
                for p in parse_formula(f, spec.var_names)] for f in (original, sliced)]
    same = all((r0 is None and r1 is None)
               or (r0 is not None and r1 is not None and r0.shape == r1.shape
                   and np.allclose(r0, r1))
               for r0, r1 in zip(*regions))
    print("same region:", same)

    if args.out:
        phi = phi_path(args.out)
        phi.parent.mkdir(parents=True, exist_ok=True)
        phi.write_text(sliced + "\n")
        print(f"wrote {phi}")


if __name__ == "__main__":
    main()
