import numpy as np
import pandas as pd
import torch
import os
from utils import projection
from parsing import interpret_ineq
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.Spec import *
from utils.readNNet import readNNet


class SimpleNN(nn.Module):
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        for w, b in zip(weights, biases):
            # print(f"Weight shape: {w.shape}, Bias shape: {b.shape}")
            # print(w[0])
            # print(b[0])
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


def read_model(model_path):
    # Load the .nnet model
    weights, biases = readNNet(model_path)
    # Create the PyTorch model
    model = SimpleNN(weights, biases)
    model.eval()
    return model


def plot_multiple_projection(plot_points, bounds, projection_direction, title=f"Exp",
                             keys=None, axis_labels=None, ax=None, point=None, sample_points=None, index=0):
    if keys is None:
        keys = [str(i) for i in range(len(plot_points))]
    if axis_labels is None:
        axis_labels = [f"X{i}" for i in range(1, len(bounds[0]) + 1)]

    colors = ['blue', 'red', 'blue', 'red', 'red', 'purple', 'brown', 'pink', 'cyan']
    line_colors = ['blue', 'yellow', 'green', 'red', 'purple', 'brown','brown', 'pink', 'cyan']

    shapes = ['o', 's', 'D', 'v', '^', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    sizes = [ 10, 6, 4, 4, 4, 4, 4, 4, 4, 4]
    alphas = [0.2, 0.3, 0.6, 0.5, 0.6, 0.7,0.9,0.55, 0.55, 0.55, 0.55, 0.55]
    line_widths = [3,1.5,1,1,0.5,4,5,6,2,2,2,1,1,1,1,1,1,1,1,1,1]
    # title = str(exp_no)
    a, b = projection_direction
    x_len = bounds[0][1] - bounds[0][0]
    y_len = bounds[1][1] - bounds[1][0]
    plt.rcParams['text.usetex'] = True
    # plt.figure(figsize=(8, 8))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax_none = True
    else:
        ax_none = False
    X_0 = [sample_points[0][i][0] for i in range(len(sample_points[0]))]
    Y_0 = [sample_points[0][i][1] for i in range(len(sample_points[0]))]
    X_1 = [sample_points[1][i][0] for i in range(len(sample_points[1]))]
    Y_1 = [sample_points[1][i][1] for i in range(len(sample_points[1]))]

    final_plots = []

    if len(X_0) > 0:
        fplt = ax.scatter(X_0[0], Y_0[0], marker='o', alpha=0.44, color='cyan', s=20, label='Low risk')
        final_plots.append(fplt)
    if len(X_1) > 0:
        fplt = ax.scatter(X_1[0], Y_1[0], marker='o', alpha=0.44, color='pink', s=20, label='High risk')
        final_plots.append(fplt)
    ax.scatter(X_1, Y_1, marker='o', alpha=0.15, color='pink', s=10,)
    ax.scatter(X_0, Y_0, marker='o', alpha=0.15, color='cyan', s=10,)

    for i, spaces in enumerate(plot_points):
        # points = spaces[0]
        j=0
        for points in spaces:
            if points is not None:
                j+=1
                ax.fill(points[:, 0], points[:, 1], alpha=alphas[i],
                          color=colors[i])
                ax.plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i])
                ax.scatter(points[:, 0], points[:, 1], marker=shapes[i],
                            s=sizes[i], color=colors[i])  # Add dots
        fplt = ax.plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i], label=keys[i])
        final_plots.append(fplt)

    fplt = ax.scatter([point[a-1]], [point[b-1]], marker='D', color='black', label="Sample point", s=20)
    final_plots.append(fplt)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(left=bounds[0][0] - 0.05*x_len , right=bounds[0][1] + 0.05*x_len)
    ax.set_ylim(bottom=bounds[1][0] - 0.05*y_len, top=bounds[1][1] + 0.05*y_len)
    ax.set_xlabel(f"X{a} ({axis_labels[a-1]})", fontsize=16)
    ax.set_ylabel(f"X{b} ({axis_labels[b-1]})", fontsize=16)
    ax.grid(alpha=0.3)

    handles = [final_plots[0], final_plots[1], final_plots[3][0], final_plots[2][0], final_plots[4]]
    labels = [h.get_label() for h in handles]
    if index == 3:
        ax.legend(handles, labels, fontsize=14, loc='lower right', bbox_to_anchor=(0.99, 0.01))
    else:
        ax.legend(handles, labels, fontsize=14)
    # try:
        # ax.legend(fontsize=14)
    # except:
    #     pass

    # Check if the directory exists
    # if not os.path.exists(f"plots/{title}"):
    #     os.makedirs(f"plots/{title}")
    # plt.savefig(f"plots/{title}/projection_{b}_{a}.png")
    # plt.show()
    # plt.close()

    # if ax_none:
    #     plt.tight_layout()
    #     if not os.path.exists(f"plots/"):
    #         os.makedirs(f"plots/")
    #     plt.savefig(f"plots/dd-exp-{title}_{b}_{a}.pdf", format='pdf')
    #     plt.show()
    #     plt.close()


def plot_compare_projections(explanations_list, vars, bounds, projection_direction=(0,1), keys=None,
                             axis_labels=[f"X{i}axis" for i in range(1, 14)], ax=None, title="exp",
                             point=None, sample_points=None, is_partitioned=None, index=0):
    a, b = projection_direction
    if point is not None:
        point, pred = point[:-1], point[-1]

    if keys is None:
        keys = [i for i in range(len(explanations_list))]
    if is_partitioned is None:
        is_partitioned = [False for i in range(len(explanations_list))]

    plot_spaces = []


    for ind, exp in enumerate(explanations_list):
        print(f"Parsing inequality from {keys[ind]}")
        parsed_formulas, parsed_vars = interpret_ineq.parse_and_interpret(exp, vars)

        if not is_partitioned[ind]:
            a,b = projection_direction
            ineq, eq = parsed_formulas[0]
            for i in range(len(vars)):
                if i not in [a, b]:
                    c = point[i]
                    condition = [0 for _ in range(len(vars) + 1)]
                    condition[i] = 1
                    condition[-1] = -1 * c
                    eq = np.append(eq, [condition], axis=0)

            parsed_formulas = [(ineq, eq)]
        assert set(vars) == set(parsed_vars)

        projected_spaces = projection.project_multi_subspace(parsed_formulas, bounds, projection_direction=projection_direction)
        plot_spaces.append(projected_spaces)

    plot_bounds = [(bounds[a][0], bounds[a][1]), (bounds[b][0], bounds[b][1])]
    plot_multiple_projection(plot_spaces, plot_bounds, (a+1,b+1), keys=keys, axis_labels=axis_labels, ax=ax, title=title, point=point, sample_points=sample_points, index=index)


def generate_samples(bounds, model, point, not_fix_points, num_samples=1000):
    samples = [[],[]]
    for i in range(num_samples):
        a, b = not_fix_points
        a_l, a_u = bounds[a]
        b_l, b_u = bounds[b]

        s_a = np.random.uniform(a_l,a_u)
        s_b = np.random.uniform(b_l,b_u)
        sample_point = point[:-1].copy()
        sample_point[a] = s_a
        sample_point[b] = s_b
        sample_point = torch.tensor(sample_point, dtype=torch.float32)
        prediction = model(sample_point)
        predicted = int(torch.round(torch.sigmoid(prediction)).squeeze().detach().numpy())
        samples[predicted].append([s_a, s_b])
    return samples



if __name__ == '__main__':
    vars = [f'x{i}' for i in range(1, 14)]
    dataset_dir = "heartAttack.csv"
    datapointss = pd.read_csv(dataset_dir)


    """ # SPECIFY: 
    # - model: path to the .nnet file
    # - exp files: the files containing the explanations
    # - exp keys: the labels shown in the legend
    # - selected_exps: the indices of the explanations to visualize
    # - selected_axis: the axes to project on, a tuple for each explanation
    """

    model = read_model("data/models/heart_attack/heart_attack-50.nnet")

    file_prefixes = [
        "examples/partial/ucore_min_itp_vars_",
        # "examples/partial/ucore_min_itp.phi.txt",
        "examples/partial/itp_vars_",
        # "examples/partial/itp.phi.txt",
        # "examples/partial/ucore_itp_vars_",
        # "examples/partial/itp_vars_",
    ]
    exp_keys = [
        # "inflated explanation",
        # "ucore-min-partial",
        # "ucore-partial",
        r'$\ensuremath{\textrm{\textbf{R}}_\mathit{min}} \circ \textrm{\textbf{Capture}}$',
        # "full-itp-ucore",
        r'$\textrm{\textbf{Capture}}$',
        # "full-itp",
    ]

    file_suffix = ".phi.txt"

    selected_axis = [(1, 5), (5, 8), (3, 12), (1, 4)]
    selected_exps = [196, 0, 0, 107]
    all_exps = []
    all_is_partitioned = []

    for ind, exp_no in enumerate(selected_exps):
        a,b = selected_axis[ind]
        explanations = []
        is_partitioned = []
        for prefix in file_prefixes:
            if "vars_" in prefix:
                file_dir = prefix + f"x{a}_x{b}" + file_suffix
                is_partitioned.append(True)
            else:
                is_partitioned.append(False)

            # print(f"Visualizing {exp_no} from {exp_file}")
            data = open(file_dir, 'rt').read()
            data = data.splitlines()
            data = [line for line in data if line.strip()]  # Remove empty lines
            exp = data[exp_no]
            explanations.append(exp)
        all_exps.append(explanations)
        all_is_partitioned.append(is_partitioned)

    plot_index = 0
    selected_axis = [(a - 1, b - 1) for a, b in selected_axis]
    row_size = 4
    col_size = len(selected_axis) // row_size
    fig, axes = plt.subplots(col_size, row_size, figsize=(5 * row_size, 5 * col_size))

    for i, explanations in enumerate(all_exps):
        exp_no = selected_exps[i]
        a,b = selected_axis[i]
        ax = axes[plot_index % row_size]

        point = datapointss.iloc[exp_no]
        sample_point = point[:-1].copy()
        sample_point = torch.tensor(sample_point, dtype=torch.float32)
        prediction = model(sample_point)
        predicted = int(torch.round(torch.sigmoid(prediction)).squeeze().detach().numpy())

        samples = generate_samples(bounds, model, point, (a, b), num_samples=12000)
        plot_compare_projections(explanations, vars, bounds,
                                 keys=exp_keys,
                                 projection_direction=(a, b),
                                 axis_labels=axis_keys,
                                 title=f"{exp_no}",
                                 ax=ax,    # Pass the subplot axis to the function
                                 point=point,
                                 sample_points=samples,
                                is_partitioned=all_is_partitioned[i],
                                 index=i
                                 )

        plot_index += 1

    plt.tight_layout()
    if not os.path.exists(f"plots/"):
        os.makedirs(f"plots/")
    plt.savefig(f"plots/0--db.svg", format='svg')
    plt.savefig(f"plots/0--db.pdf", format='pdf')
    plt.show()
    plt.close()

