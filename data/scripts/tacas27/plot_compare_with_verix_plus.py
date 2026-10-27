# Compares our experiments with the VeriX+ ones in one figure per model family, one subplot per task.
# USAGE: plot_compare_with_verix_plus.py <stats_dir> [<output_dir>]
#   <stats_dir>   directory with our <task>.txt stats files (of `runme.sh collect-stats`), e.g. stats/short;
#                 read as in plot_fc_stats.py, also their timeouts per sample (timeout lines and y-limits)
#   <output_dir>  the output directory of plot_fc_stats.py (default: <stats_dir>), e.g. data/output/short:
#                 it holds the VeriX+ CSVs fig-8_comparison-plots_<task><suffix>_verix.csv (of `runme.sh
#                 plot-performance`; <suffix> is '' for the 50x models and '200' for the 200x ones), and gets the
#                 PDF plots fig-8_comparison-plots_PAR-2_ours_vs_verix_plus_<layout>_<family>.pdf
import os
import shutil
import sys
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ploting_utils import *
import matplotlib as mpl

if len(sys.argv) not in (2, 3):
    sys.exit(f'USAGE: {sys.argv[0]} <stats_dir> [<output_dir>]')
ours_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else ours_dir
# The VeriX+ CSVs are next to the CSVs of plot_fc_stats.py
verix_dir = output_dir
os.makedirs(output_dir, exist_ok=True)
# Prefix of the names of the CSV and PDF files, as in plot_fc_stats.py
output_prefix = 'fig-8_comparison-plots_'

# Full LaTeX rendering needs a LaTeX installation; without one, Matplotlib's mathtext is used
use_tex = shutil.which('latex') is not None

# Keep text rendering consistent across scripts:
mpl.rcParams.update({
    'text.usetex': use_tex,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
})

image_tasks = ['mnist', 'cifar', 'gtsrb']
tabular_tasks = ['heart_attack', 'obesity', 'loan']
widths = ['50', '200']
# Upper y-limit (PAR-2, s) of the tabular subplots per width: they are fast, so the plots zoom in
tabular_ylims = {'50': 30, '200': 280}


def ylim(task, width):
    # Upper y-limit (PAR-2, s) of a subplot: image tasks sit 5 s above 2 * timeout so the timed-out points stay
    # visible; the tabular ones zoom in (tabular_ylims), but not above 2 * timeout + 2 s
    if task in tabular_tasks:
        return min(tabular_ylims[width], 2 * timeouts[task] + 2)
    return 2 * timeouts[task] + 5


# Model family -> (our 'Width' value, VeriX+ CSV suffix)
families = {'50': ('50', ''), '200': ('200', '200')}
# Figure layouts to produce: name -> (tasks, #rows of the subplot grid); tasks fill the grid row by row
layouts = {
    'all_tasks': (tabular_tasks + image_tasks, 2),
    'image_only': (image_tasks, 1),
}

# Ours: which methods and Itp algorithms to show (all 4 x 5 lines would bury the VeriX+ ones)
ours_methods = [rename_single_methods('naive'), 'Guide-all', 'Fix-all', 'Hybrid']
ours_itp = ['weak']  # any of 'weaker', 'weak', 'mid', 'strong', 'stronger'
# Only the model depths in shown_depths(task) (ploting_utils.py) are plotted

# VeriX+: CSV columns are Algorithm, epsilon, layers (the model depth, from 1), PAR2 (as write_verix_csvs of
# runme.sh). Only the IR algorithm is shown, i.e. the random traversal order, as run-verix-plus (the original
# experiments); the legend does not name it. Drawn dashed, one color per epsilon (verix_line_style in ploting_utils.py).
verix_algorithm = 'IR'
verix_name = 'VeriX+'
epsilon_order = ['100%', '5%', '0.5%']
epsilon_keys = {'100%': 'e=1', '5%': 'e=0.05', '0.5%': 'e=0.005'}


def read_ours(task, width):
    path = os.path.join(ours_dir, f'{task}.txt')
    if not os.path.exists(path):
        print(f"Warning: {path} not found, no data of ours for {task}")
        return None
    data, task_timeouts = read_stats(path, task)
    # draw_timeout_lines and ylim read the timeouts of the stats
    use_stats_timeouts(task, task_timeouts)
    data = data[data['Width'] == width].copy()
    data = rename_methods(data)
    data = rename_itp(data)
    data['Model Size'] = pd.to_numeric(data['Model Size'], errors='coerce')
    data['PAR-2'] = pd.to_numeric(data['PAR-2'], errors='coerce')
    data = data[data['Method'].isin(ours_methods) & data['Experiment Name'].isin(ours_itp)
                & data['Model Size'].isin(shown_depths(task))].copy()
    data['_method_rank'] = data['Method'].map({m: i for i, m in enumerate(ours_methods)})
    data['_itp_rank'] = data['Experiment Name'].map({a: i for i, a in enumerate(ours_itp)}).astype(int)
    return data.sort_values(by=['_method_rank', '_itp_rank', 'Model Size']) if len(data) else None


def read_verix(task, suffix):
    path = os.path.join(verix_dir, f'{output_prefix}{task}{suffix}_verix.csv')
    if not os.path.exists(path):
        print(f"Warning: {path} not found, no VeriX+ data for {task}")
        return None
    data = pd.read_csv(path)
    data['PAR2'] = pd.to_numeric(data['PAR2'], errors='coerce')
    data = data[(data['Algorithm'] == verix_algorithm)
                & pd.to_numeric(data['layers'], errors='coerce').isin(shown_depths(task))].copy()
    if not len(data):
        return None
    data['_eps_rank'] = data['epsilon'].map({e: i for i, e in enumerate(epsilon_order)})
    return data.sort_values(by=['_eps_rank', 'layers'])


# Only the all_tasks layout is plotted, for every model family
shown_layouts = ['all_tasks']
for (layout, (tasks, n_rows)), (family, (width, suffix)) in (
        ((l, layouts[l]), f) for l in shown_layouts for f in families.items()):
    ncols = int(np.ceil(len(tasks) / n_rows))
    fig, axes = plt.subplots(n_rows, ncols, figsize=(6 * ncols, 6 * n_rows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[len(tasks):]:  # empty cells when the tasks do not fill the grid
        ax.set_axis_off()

    for ax, task in zip(axes, tasks):
        ours = read_ours(task, width)
        verix = read_verix(task, suffix)
        ax.set_title(f'{rename_task_name(task)}', fontsize=title_fs)
        if ours is None and verix is None:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center', fontsize=label_fs)
            ax.set_axis_off()
            continue

        layers = set()  # model depths, for the top x-axis
        n_relus = int(width)  # ReLUs per hidden layer
        if ours is not None:
            for (method, algorithm), group in ours.groupby(['Method', 'Experiment Name'], sort=False, observed=True):
                ax.plot(group['Model Size'] * n_relus, group['PAR-2'], label=rename_method_algorithm(method, algorithm),
                        **our_line_style(method, algorithm, len(ours_itp) > 1))
            layers |= set(ours['Model Size'].dropna().astype(int))

        if verix is not None:
            for epsilon, group in verix.groupby('epsilon', sort=False):
                label = rename_method_algorithm(verix_name, epsilon_keys.get(epsilon, epsilon))
                ax.plot(group['layers'] * n_relus, group['PAR2'], label=label, **verix_line_style(epsilon))
            layers |= set(verix['layers'].dropna().astype(int))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'\#ReLUs' if use_tex else '#ReLUs', fontsize=label_fs)
        ax.set_ylabel('Average PAR-2 (s)', fontsize=label_fs)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        # Model names (<width>x<#hidden layers>) on a top x-axis
        draw_model_names(ax, [(f'{width}x{d}', d * n_relus) for d in layers])
        ax.set_ylim(bottom=0, top=ylim(task, width))
        draw_timeout_lines(ax, task)
        ax.grid(True)

    # Collect legend entries from all subplots (some lines may be missing on some tasks)
    legend_entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_entries.setdefault(label, handle)

    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        title='Method - ' + r'$\mathit{Itp}$ Algorithm / $\epsilon$',
        loc='upper center',
        title_fontsize=legend_fs + 2,
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        columnspacing=0.8,
        fontsize=legend_fs,
        frameon=False
    )
    fig.suptitle(f'Model family {family}', fontsize=title_fs + 2)

    plt.tight_layout()
    plot_name = os.path.join(output_dir, f'{output_prefix}PAR-2_ours_vs_verix_plus_{layout}_{family}.pdf')
    plt.savefig(plot_name, format='pdf', bbox_inches='tight')
    print(f"Saved runtime comparison plot to {plot_name}")

# plt.show()
