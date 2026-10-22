"""Plot the 2D snapshot of one sample's explanation(s) on a chosen axis pair.

Run from the repo root:

    PYTHONPATH=. python3 python_scripts/visualization/plot_diameter.py \
        --index 71 \
        --exp data/explanations/heart_attack/50x1/s100_scaled/fix-all/itp_aweak_bstrong.phi.txt \
        --model data/models/heart_attack/heart_attack_50x1.nnet \
        --x x10 --y x1 \
        --out plots/ha_s71_oldpeak_age \
        --slice plots/ha_s71_sliced_x10_x1.phi.txt \
        --domain data/domain/heart_attack/ranges.csv

Several explanation files can be overlaid by repeating them after --exp.
"""

import argparse
from pathlib import Path

from explviz import (Classifier, Explanation, Panel, load_points,
                     load_explanations, plot_grid, save_figure)
from explviz.paths import data_path
from explviz.specs import HEART_ATTACK_SCALED as spec

import sys
import runpy
import csv

def feature_index(s):
    """'x10' -> 9 (1-based variable name); '9' -> 9 (0-based index)."""
    return int(s[1:]) - 1 if s.startswith("x") else int(s)

def read_csv(file_path):
    min_values = []
    max_values = []
    feature_names = []
    the_variables = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            try:
                min_values.append(float(row['Min']))
                max_values.append(float(row['Max']))
                feature_names.append(row['Feature'])
                the_variables.append(row['Var'])
                
            except ValueError:
                print("Error")
                
    return min_values, max_values, the_variables, feature_names

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", "-i", type=int, required=True,
                   help="sample index (= line index in the .phi.txt)")
    p.add_argument("--exp", nargs="+", required=True,
                   help="explanation .phi.txt file(s)")
    p.add_argument("--label", nargs="+",
                   help="legend label per --exp file (default: the method folder name)")
    p.add_argument("--model", required=True, help=".nnet model file")
    p.add_argument("--x", required=True, help="x-axis feature: 'x10' or 0-based index '9'")
    p.add_argument("--y", required=True, help="y-axis feature: 'x1' or 0-based index '0'")
    p.add_argument("--out", "-o", required=True,
                   help="output path without suffix (writes .png and .pdf)")
    p.add_argument("--slice",required=True,help="output of slice_explanation_formula")
    p.add_argument("--data",
                   default=str(data_path("datasets/heart_attack/heart_attack_s100_scaled.csv",
                                         must_exist=False)),
                   help="dataset CSV the sample index refers to")
    p.add_argument("--domain",required=False,help="Feature domain ranges")
    return p.parse_args()


def get_diameter_formula(my_file):
    formula = ""
    with open(my_file,"r") as g:
        for line in g:
            formula+=str(line.rstrip())
    return formula

def main():
    args = parse_args()
    if args.label and len(args.label) != len(args.exp):
        raise SystemExit(f"got {len(args.exp)} --exp files but {len(args.label)} --label values")
    labels = args.label or [Path(f).parent.name for f in args.exp]

    target_file1 = Path("diameters/compute_hypervolume.py")
    target_dir1 = str(target_file1.parent)
    original_argv = sys.argv.copy()
    original_path = sys.path.copy()
    slice_x = str(args.x)+"=0,1"
    slice_y = str(args.y)+"=0,1"
    sys.argv = [str(target_file1),"--filename",args.slice,"--slices",slice_x,slice_y,"--out","plots/plotting_diameter"]
    sys.path.insert(0, target_dir1)
    try:
        runpy.run_path(target_file1, run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path = original_path

    min_values = []
    max_values = []
    the_variables = []
    feature_names = []
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    if (args.domain):
        min_values, max_values, the_variables, feature_names = read_csv(args.domain)
        for i in range(len(the_variables)):
            if (the_variables[i]==args.x):
                x_min = min_values[i]
                x_max = max_values[i]
            if (the_variables[i]==args.y):
                y_min = min_values[i]
                y_max = max_values[i]
        
    X, _ = load_points(args.data)
    clf = Classifier.from_nnet(args.model)
    i = args.index

    explanations = []
    for path, label in zip(args.exp, labels):
        print(label)
        formulas = load_explanations(path)
        if not 0 <= i < len(formulas):
            raise SystemExit(f"{path} has {len(formulas)} formulas; index {i} is out of range")
        explanations.append(Explanation(formulas[i], label, label))

    explanations.append(Explanation(get_diameter_formula("plots/plotting_diameter.smt2"),"diameter","diameter"))
    panel = Panel(
        point=X[i],
        axes=(feature_index(args.x), feature_index(args.y)),
        explanations=explanations,
        mode="snapshot",
    )
    fig = plot_grid([panel], x_min, x_max, y_min, y_max, spec, ncols=1, clf=clf)
    for written in save_figure(fig, args.out):
        print(written)


if __name__ == "__main__":
    main()
