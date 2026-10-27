# Plots of the CNN (ONNX) stats files of one runme.sh mode, i.e. of `runme.sh collect-stats <mode>`.
# USAGE: plot_cnn_stats.py <stats_dir> [<output_dir>]
#   <stats_dir>   directory with the <task>_cnn.txt stats files, e.g. stats/short
#   <output_dir>  directory for the CSV tables of the stats and the PDF plots (default: <stats_dir>),
#                 named performance-plots_<task>_cnn.csv and performance-plots_<plot>.pdf
# One plot per CNN model family, PAR-2 over the #ReLUs of the models (from data/models/<task>/cnn-bench/manifest.csv).
# Tasks without a stats file and families without results are left out.
import os
import re
import shutil
import sys
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from ploting_utils import *
import matplotlib as mpl

if len(sys.argv) not in (2, 3):
    sys.exit(f'USAGE: {sys.argv[0]} <stats_dir> [<output_dir>]')
stats_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else stats_dir
os.makedirs(output_dir, exist_ok=True)
# Prefix of the names of the CSV and PDF files
output_prefix = 'performance-plots_'

# Full LaTeX rendering needs a LaTeX installation; without one, Matplotlib's mathtext is used
use_tex = shutil.which('latex') is not None

# Keep text rendering consistent across scripts:
mpl.rcParams.update({
    'text.usetex': use_tex,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
})

here = os.path.dirname(os.path.abspath(__file__))
tasks = ['mnist', 'cifar', 'gtsrb']


def ylim(task):
    # Upper y-limit (PAR-2, s) of a subplot: 5 s above 2 * timeout, so that the timed-out points stay visible
    return 2 * timeouts[task] + 5


plot_case = 'PAR-2'  # Options: 'Size [MB]', 'Time', 'PAR-2', 'completed', 'terms'

# Which Itp algorithms to show: any of 'weaker', 'weak', 'mid', 'strong', 'stronger'
shown_itp = ['weak']

# Model groups: each group gets its own plot. Within a group, models are ordered by prefix
# (in the listed order), then by number, e.g. NP1, NP2, ..., P1, P2, ...
model_groups = {
    'avg-pool': ['AB'],
    # 'narrow-avg-pool': ['NA'],
    'narrow-avg-pool': ['NAB'],
    'no-pool': ['S'],
    'narrow-no-pool': ['NS'],
    # 'max-pool': ['NP', 'P'],
}


# Number of ReLU neurons per model, read from each task's cnn-bench manifest (differs per dataset)
models_dir = os.path.join(here, '..', '..', 'models')


def read_relu_counts(task):
    manifest = pd.read_csv(os.path.join(models_dir, task, 'cnn-bench', 'manifest.csv'))
    return dict(zip(manifest['id'], manifest['relu_neurons']))


def model_prefix(model):
    return re.match(r'^[A-Za-z]+', model).group(0)


def read_task(task):
    input_file = os.path.join(stats_dir, f'{task}_cnn.txt')
    output_file = os.path.join(output_dir, f'{output_prefix}{task}_cnn.csv')
    if not os.path.exists(input_file):
        print(f'No CNN stats of {task}: {input_file}')
        return None
    df, task_timeouts = read_stats(input_file, task, cnn=True)

    df.to_csv(output_file, index=False)
    print(f"CSV file saved to {output_file}")

    # ylim and draw_timeout_lines read the timeouts of the stats
    use_stats_timeouts(task, task_timeouts)

    data = pd.read_csv(output_file)
    data = rename_methods(data)
    data = rename_itp(data)
    data['terms'] = data['#Terms']
    data['completed'] = data['%Completed']
    data['Time'] = data['Time [s]']
    return data


task_data = {task: data for task in tasks if (data := read_task(task)) is not None}
relu_counts = {task: read_relu_counts(task) for task in task_data}
method_order = [rename_single_methods('naive'), 'Guide-all', 'Fix-all', 'Hybrid']
rank = {m: i for i, m in enumerate(method_order)}

for group_name, prefixes in model_groups.items():
    # Only the tasks with results for the models of the group
    shown_tasks = [t for t in tasks if t in task_data and task_data[t]['Model Size'].map(model_prefix).isin(prefixes).any()]
    if not shown_tasks:
        print(f'Skipping {group_name}: no results')
        continue

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6))
    axes = np.atleast_1d(axes)
    print(f'Plotting {group_name}')

    for ax, task in zip(axes, tasks):
        # A task without results keeps its place in the row, but empty
        if task not in shown_tasks:
            ax.set_visible(False)
            continue

        data = task_data[task]
        data = data[data['Model Size'].map(model_prefix).isin(prefixes) & data['Experiment Name'].isin(shown_itp)].copy()

        # x-axis is the number of ReLU neurons of the model
        data['x'] = data['Model Size'].map(relu_counts[task])
        missing = sorted(data.loc[data['x'].isna(), 'Model Size'].unique())
        if missing:
            print(f"Warning: no relu_neurons in {task} manifest for {missing}, skipping them")
            data = data.dropna(subset=['x'])

        data['_method_rank'] = data['Method'].map(rank).fillna(len(method_order)).astype(int)
        data = data.sort_values(by=['_method_rank', 'Method', 'Experiment Name', 'x'])

        for (method, algorithm), group in data.groupby(['Method', 'Experiment Name'], sort=False, observed=True):
            ax.plot(group['x'], group[plot_case], label=rename_method_algorithm(method, algorithm),
                    **our_line_style(method, algorithm, len(shown_itp) > 1))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'\#ReLUs' if use_tex else '#ReLUs', fontsize=label_fs)
        ax.set_ylabel('Average PAR-2 (s)' if plot_case == 'PAR-2' else plot_case, fontsize=label_fs)
        ax.set_title(f'{rename_task_name(task)}', fontsize=title_fs)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        # Model names once, on a top x-axis at each model's #ReLUs
        draw_model_names(ax, data.drop_duplicates('Model Size')[['Model Size', 'x']].itertuples(index=False))
        if plot_case == 'PAR-2':
            ax.set_ylim(0, ylim(task))
            draw_timeout_lines(ax, task)
        ax.grid(True)

    # Collect legend entries from all subplots (some methods may be missing on some tasks)
    legend_entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_entries.setdefault(label, handle)

    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        title='Method - ' + r'$\mathit{Itp}$ Algorithm',
        loc='upper center',
        title_fontsize=legend_fs + 2,
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        columnspacing=0.8,
        fontsize=legend_fs,
        frameon=False
    )
    fig.suptitle(cnn_family_title(prefixes[0]) if len(prefixes) == 1 else group_name, fontsize=title_fs + 2)

    plt.tight_layout()
    plot_name = os.path.join(output_dir, f'{output_prefix}{plot_case}_cnn_relus_{group_name}.pdf')
    plt.savefig(plot_name, format='pdf', bbox_inches='tight')
    print(f"Saved runtime comparison plot to {plot_name}")

# plt.show()
