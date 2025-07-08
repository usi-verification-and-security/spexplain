import os

from matplotlib import pyplot as plt

from utils import projection
# import parse_smt2
from parsing import interpret_ineq
from utils.Spec import *



def plot_multiple_projection(plot_points, bounds, projection_direction, title=f"Exp", keys=None, axis_labels=None, ax=None, index=0):
    if keys is None:
        keys = [str(i) for i in range(len(plot_points))]
    if axis_labels is None:
        axis_labels = [f"X{i}" for i in range(1, len(bounds[0]) + 1)]

    colors = ['yellow', 'orange', 'green', 'blue', 'red', 'purple', 'brown', 'pink', 'cyan']
    line_colors = ['green', 'blue', 'orange', 'yellow', 'green', 'blue', 'red', 'purple', 'brown', 'pink', 'cyan']

    shapes = ['o', 's', 'D', 'v', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    sizes = [9, 9, 8, 8, 14, 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    alphas = [ 0.12, 0.5, 0.7, 0.7,  1, 0.7,0.9,0.55, 0.55, 0.55, 0.55, 0.55]
    line_widths = [3,2,2,2,2,5,6,2,2,2,1,1,1,1,1,1,1,1,1,1]
    # title = str(exp_no)
    a, b = projection_direction
    x_len = bounds[0][1] - bounds[0][0]
    y_len = bounds[1][1] - bounds[1][0]
    # plt.figure(figsize=(8, 8))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax_none = True
    else:
        ax_none = False

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
        ax.plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i], label=keys[i])
    # ax.fill(points[:, 0], points[:, 1], alpha=alphas[i],
    #         color=colors[i])
    # ax.plot(points[:, 0], points[:, 1], linewidth=line_widths[i], color=colors[i])
    # ax.scatter(points[:, 0], points[:, 1], marker=shapes[i],
    #            s=sizes[i], color=colors[i])  # Add dots
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(left=bounds[0][0] - 0.05*x_len , right=bounds[0][1] + 0.05*x_len)
    ax.set_ylim(bottom=bounds[1][0] - 0.05*y_len, top=bounds[1][1] + 0.05*y_len)
    ax.set_xlabel(f"X{a} ({axis_labels[a-1]})", fontsize=16)
    ax.set_ylabel(f"X{b} ({axis_labels[b-1]})", fontsize=16)
    ax.grid(alpha=0.3)
    try:
        if index in [1,2]:
            ax.legend(loc='lower right', bbox_to_anchor=(0.9, 0.1), fontsize=14)
        else:
            ax.legend(fontsize=14)
    except:
        pass

    # Check if the directory exists
    # if not os.path.exists(f"plots/{title}"):
    #     os.makedirs(f"plots/{title}")
    # plt.savefig(f"plots/{title}/projection_{b}_{a}.png")
    # plt.show()
    # plt.close()
    if ax_none:
        plt.tight_layout()
        if not os.path.exists(f"plots/{title}"):
            os.makedirs(f"plots/{title}")
        plt.savefig(f"plots/{title}/00-projection_{a}_{b}.pdf", format='pdf')
        # plt.show()
        plt.close()


def plot_compare_projections(explanations_list, vars, bounds, projection_direction=(0,1), keys=None,
                             axis_labels=[f"X{i}axis" for i in range(1, 14)], ax=None, title="exp", index=0):
    if keys is None:
        keys = [i for i in range(len(explanations_list))]

    plot_spaces = []
    a, b = projection_direction

    for ind, exp in enumerate(explanations_list):
        print(f"Parsing inequality from {keys[ind]}")
        parsed_formulas, parsed_vars = interpret_ineq.parse_and_interpret(exp, vars)

        assert set(vars) == set(parsed_vars)

        projected_spaces = projection.project_multi_subspace(parsed_formulas, bounds, projection_direction=projection_direction)
        plot_spaces.append(projected_spaces)

    plot_bounds = [(bounds[a][0], bounds[a][1]), (bounds[b][0], bounds[b][1])]
    plot_multiple_projection(plot_spaces, plot_bounds, (a+1,b+1), keys=keys, axis_labels=axis_labels, ax=ax, title=title, index=index)


if __name__ == '__main__':
    vars = [f'x{i}' for i in range(1, 14)]


    """ # SPECIFY: 
    # - exp files: the files containing the explanations
    # - exp keys: the labels shown in the legend
    # - selected_exps: the indices of the explanations to visualize
    # - selected_axis: the axes to project on, a tuple for each explanation
    """
    exp_files = [
        "explanations/itp_aweak_bweak.phi.txt",
        "explanations/itp_aweak_bstrong.phi.txt",
        # "explanations/itp_afactor_0.5_bweak.phi.txt",
        "explanations/itp_afactor_0.5_bstrong.phi.txt",
        # "explanations/itp_astrong_bweak.phi.txt", #full_page, not convex
        "explanations/itp_astrong_bstrong.phi.txt",
        # "explanations/itp_astronger_bweak.phi.txt", #Single dot
        "explanations/itp_astronger_bstrong.phi.txt",  # Single dot
    ]
    exp_keys = [
        r'weaker',
        r'weak',
        r'mid',
        r'strong',
        r'stronger',
    ]
    selected_exps = [196, 0, 0, 107]
    selected_axis = [(1, 5), (5, 8), (3, 12), (1, 4)]

    all_exps = []
    for exp_no in selected_exps:
        explanations = []
        for exp_file in exp_files:
            print(f"Visualizing {exp_no} from {exp_file}")
            data = open(exp_file, 'rt').read()
            data = data.splitlines()
            data = [line for line in data if line.strip()]  # Remove empty lines
            exp = data[exp_no]
            explanations.append(exp)
        all_exps.append(explanations)

    plot_index = 0
    selected_axis = [(a - 1, b - 1) for a, b in selected_axis]
    row_size = 4
    col_size = len(selected_axis) // row_size
    fig, axes = plt.subplots(col_size, row_size, figsize=(5 * row_size, 5 * col_size))

    for i, explanations in enumerate(all_exps):
        exp_no = selected_exps[i]
        a,b = selected_axis[i]
        ax = axes[plot_index % row_size]

        plot_compare_projections(explanations, vars, bounds,
                                 keys=exp_keys,
                                 projection_direction=(a, b),
                                 axis_labels=axis_keys,
                                 title=f"{selected_exps[i]}",
                                 ax=ax,  # Pass the subplot axis to the function
                                 index=i
                                 )
        plot_index += 1

    plt.tight_layout()
    if not os.path.exists(f"plots/"):
        os.makedirs(f"plots/")
    plt.savefig(f"plots/0--itps.pdf", format='pdf')
    plt.savefig(f"plots/0--itps.svg", format='svg')
    plt.show()
    plt.close()

