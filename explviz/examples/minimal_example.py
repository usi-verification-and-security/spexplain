"""The smallest useful explviz script: one explanation, one axis pair.

    PYTHONPATH=. python3 -m explviz.examples.minimal_example
"""

from explviz import (Classifier, Explanation, Panel, load_points,
                     load_explanations, plot_grid, save_figure)
from explviz.paths import data_path, explanations_path
from explviz.specs import HEART_ATTACK_SCALED as spec

X, _ = load_points(data_path("datasets/heart_attack_s100_scaled.csv"))
formulas = load_explanations(
    explanations_path("HA/50x1/fix-all/itp_aweak_bstrong.phi.txt"))
clf = Classifier.from_nnet(data_path("models/heart_attack_50x1.nnet"))

panel = Panel(
    point=X[71],
    axes=(9, 0),                       # 0-based feature indices: Oldpeak vs Age
    explanations=[Explanation(formulas[71], "Fix-all", "fix-all")],
    mode="projection",       # a projection draws no class regions
)
fig = plot_grid([panel], spec, ncols=1, clf=clf)
save_figure(fig, "plots/explviz/demo")
