# Plots of the stats files of one runme.sh mode, i.e. of `runme.sh collect-stats <mode>`.
# USAGE: plot_fc_stats.py <stats_dir> [<plots_dir>]
#   <stats_dir>  directory with the <task>.txt stats files, e.g. stats/short
#   <plots_dir>  directory for the PDF plots (default: <stats_dir>/plots)
# The CSV tables of the stats go to <stats_dir>/<task>.csv.
# Tasks without a stats file and widths without results are left out.
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
    sys.exit(f'USAGE: {sys.argv[0]} <stats_dir> [<plots_dir>]')
stats_dir = os.path.abspath(sys.argv[1])
plots_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else os.path.join(stats_dir, 'plots')

# Full LaTeX rendering needs a LaTeX installation; without one, Matplotlib's mathtext is used
use_tex = shutil.which('latex') is not None

# Keep text rendering consistent across scripts:
mpl.rcParams.update({
    'text.usetex': use_tex,            # False uses Matplotlib's mathtext (recommended unless you need full LaTeX)
    'font.family': 'sans-serif',        # choose 'serif' or 'sans-serif' to match your desired look
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
})

# Each dataset file contains the results for all model widths (e.g. 50xN and 200xN)
image_tasks = ['mnist', 'cifar', 'gtsrb']
tabular_tasks = ['heart_attack', 'obesity', 'loan']
tasks = image_tasks + tabular_tasks
# Figure layouts to produce: name -> (tasks, grid shape, figure size); image tasks fill the top row
layouts = {
    'all_tasks': (image_tasks + tabular_tasks, (2, 3), (15.1, 12.5)),
    'image_only': (image_tasks, (1, 3), (15.1, 4.8)),
}
widths = ['50', '200']


def ylim(task, width):
    # Upper y-limit (PAR-2, s) of a subplot: 2 * timeout plus a margin, so that the timed-out points stay visible
    # (image tasks 5 s, tabular 2 s above); the tabular 50x models are fast, so their plots zoom in to 30 s
    if task in tabular_tasks and width == '50':
        return 30
    return 2 * timeouts[task] + (5 if task in image_tasks else 2)


# Itp algorithms to plot per model width; widths not listed show all of them
# (any of 'weaker', 'weak', 'mid', 'strong', 'stronger')
shown_itp = {'200': ['weak']}
# Draw missing data points at PAR-2 = 2 * timeout (see timeouts in ploting_utils.py)
fill_missing_points = True
# filter_experimnet = 'astrong_bstrong'
plot_case = 'PAR-2'  # Options: 'Size [MB]', 'Time', 'PAR-2', 'completed', 'terms'
# plot_case = 'Size [MB]'
# Read each dataset file once; it contains the results of all model widths
all_data = {}

for task in tasks:
    input_file = os.path.join(stats_dir, f'{task}.txt')
    output_file = os.path.join(stats_dir, f'{task}.csv')
    if not os.path.exists(input_file):
        print(f'No stats of {task}: {input_file}')
        continue
    # plot_name = f'plots/{task}_runtime_comparison.pdf'
    # Initialize a list to store table rows
    rows = []

    # Regular expression to match table rows
    # NOTE: new column "size [MB]" was added to the table; in the current output it appears after '#terms' and before '#checks'.
    # The regex captures the experiment name plus 10 numeric/text fields in the order shown in the table.
    table_row_pattern = re.compile(
        r'^\s*(\S.*?)\s+\|\s+([\d.X%]+)\s+\|\s+([\d.X%]+)\s+\|\s+([\d.X%]+)\s+\|\s+([\d.X]+)\s+\|\s+([\d.X]+)\s+\|\s+([\d.X]+)\s+\|\s+([\d.X%]+)\s+\|\s+([\d.X]+)\s+\|\s+([\d.X]+)')

    # The timeout per sample of the mode, as written by collect_stats.sh
    timeout_pattern = re.compile(r'^Timeout per sample \[s\]: *([0-9.]+)')
    task_timeouts = set()

    # Read the file and process it
    with open(input_file, 'r') as file:
        current_experiment = None
        header_pattern = re.compile(rf'^{re.escape(task)}\s+\d+x\d+\s+\S+')
        for line in file:
            line = line.strip()
            if timeout_pattern.match(line):
                task_timeouts.add(float(timeout_pattern.match(line).group(1)))
            elif header_pattern.match(line):  # Experiment name, e.g. 'loan 50x4 fix-all'
                current_experiment = line
                parts = current_experiment.split()
                width, model_size = parts[1].split('x')  # e.g. 50x4 -> width 50, 4 hidden layers
                method = parts[-1]  # Extract the last word
            elif current_experiment and table_row_pattern.match(line):  # Match table rows
                match = table_row_pattern.match(line)
                # Extract row data
                row = [current_experiment, width, model_size, method] + list(match.groups())
                # Check if `%completed` is not `X`
                # Note: with the added 'Width' and 'Size [MB]' columns the index of '%Completed' is now 11.
                if row[11] != 'X':  # `%completed` column (index 11 in the row)
                    # Replace `X` with NaN for other columns
                    row = [np.nan if value == 'X' else value for value in row]
                    rows.append(row)

    # Create a DataFrame from the rows
    # The captured order (after Experiment/Width/Model/Method) is:
    # Experiment Name, %Features, %Fixed, %Dimension, #Terms, Size [MB], #Checks, %Completed, Time [s], PAR-2
    columns = ['Experiment', 'Width', 'Model Size', 'Method', 'Experiment Name', '%Features', '%Fixed', '%Dimension',
               '#Terms', 'Size [MB]', '#Checks', '%Completed', 'Time [s]', 'PAR-2']
    df = pd.DataFrame(rows, columns=columns)
    # Some stats files contain the same experiment blocks more than once
    df = df.drop_duplicates(subset=['Width', 'Model Size', 'Method', 'Experiment Name'], keep='first')

    percent_columns = ['%Features', '%Fixed', '%Dimension', '%Completed']
    for col in percent_columns:
        if col in df.columns:
            df[col] = df[col].str.replace('%', '', regex=False).astype(float)

    # Convert other numeric columns to appropriate types (include new 'Size [MB]')
    numeric_columns = ['#Terms', '#Checks', 'Size [MB]', 'Time [s]', 'PAR-2']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"CSV file saved to {output_file}")

    all_data[task] = pd.read_csv(output_file, dtype={'Width': str})

    # The timeouts of the modes differ (see modes.sh), so use those of the stats instead of the defaults of
    # ploting_utils.py; fill_missing and draw_timeout_lines read the same dictionary
    if task_timeouts:
        if len(task_timeouts) > 1:
            print(f'WARNING: {task} has several timeouts per sample: {sorted(task_timeouts)}; using the largest one')
        timeouts[task] = max(task_timeouts)

for (layout, (layout_tasks, grid, figsize)), name in [(l, w) for l in layouts.items() for w in widths]:
    # Only the tasks with results for the current width
    shown_tasks = [t for t in layout_tasks if t in all_data and (all_data[t]['Width'] == name).any()]
    if not shown_tasks:
        print(f'Skipping {layout} for model width {name}: no results')
        continue

    # Ploting the runtime vs model size; the legend goes below the grid
    fig, axes = plt.subplots(*grid, figsize=figsize)
    print(f'Plotting {layout} for model width {name}')
    for ax, task in zip(axes.flatten(), layout_tasks):
        # A task without results keeps its place in the grid, but empty
        if task not in shown_tasks:
            ax.set_visible(False)
            continue

        # Select only the models of the current width
        data = all_data[task]
        data = data[data['Width'] == name].copy()
        # Without the unsat cores of interpolants of runme.sh (heart_attack 50x1 only)
        data = data[~data['Experiment Name'].str.contains('ucore_')]
        # if '200' in task:
            # data = data[data['Experiment Name'].str.contains(filter_experimnet)]

        # Ensure model_size and runtime are numeric
        data = rename_methods(data)
        data = rename_itp(data)
        data['Model Size'] = pd.to_numeric(data['Model Size'], errors='coerce')
        data['PAR-2'] = pd.to_numeric(data['PAR-2'], errors='coerce')
        data['terms'] = pd.to_numeric(data['#Terms'], errors='coerce')
        data['completed'] = pd.to_numeric(data['%Completed'], errors='coerce')
        data['Time'] = pd.to_numeric(data['Time [s]'], errors='coerce')
        data['Size [MB]'] = pd.to_numeric(data['Size [MB]'], errors='coerce')
        data = data[data['Model Size'].isin(shown_depths(task))]
        if name in shown_itp:
            data = data[data['Experiment Name'].isin(shown_itp[name])]
        if fill_missing_points and plot_case == 'PAR-2':
            data = fill_missing(data, task, ['Method', 'Experiment Name'], 'Model Size', 'PAR-2', shown_depths(task))

        # x-axis is the number of ReLU neurons: every hidden layer has <width> ReLUs
        data['x'] = data['Model Size'] * int(name)

        # sort data by Model Size
        data = data.sort_values(by='Model Size')

        # Group by both Method and Algorithm and plot
        # plt.figure(figsize=(10, 6))
        method_order = [rename_single_methods('naive'), 'Guide-all', 'Fix-all', 'Hybrid']
        rank = {m: i for i, m in enumerate(method_order)}
        data['_method_rank'] = data['Method'].map(rank).fillna(len(method_order)).astype(int)
        data = data.sort_values(by=['_method_rank', 'Method', 'Experiment Name', 'Model Size'])
        for (method, algorithm), group in data.groupby(['Method', 'Experiment Name'], sort=False, observed=True):
            # ax.plot(group['Model Size'], group['terms'], label=f"{method} - {algorithm}", color=color, marker=marker, linestyle='-', linewidth=1)

            ax.plot(group['x'], group[plot_case], label=rename_method_algorithm(method, algorithm),
                    **our_line_style(method, algorithm, len(shown_itp.get(name, itp_markers)) > 1))


        # Add labels, legend, and title
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.rcParams['text.usetex'] = use_tex


        ax.set_xlabel(r'\#ReLUs' if use_tex else '#ReLUs', fontsize=label_fs)
        ax.set_ylabel('Average PAR-2 (s)' if plot_case == 'PAR-2' else plot_case, fontsize=label_fs)
        ax.set_title(f'{rename_task_name(task)}', fontsize=title_fs)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        # Model names (<width>x<#hidden layers>) on a top x-axis
        draw_model_names(ax, [(f'{name}x{d}', d * int(name)) for d in shown_depths(task)])

        if plot_case =='PAR-2':
            ax.set_ylim(0, ylim(task, name))

            if task.__contains__('gtsrb'):
                #     set y-axis lables with 200 separation
                ax.set_yticks(np.arange(0, ylim(task, name) + 1, 200))
            draw_timeout_lines(ax, task)
        # plt.legend(title='Method - Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

    # The legend handles and labels of all subplots, each once (the first subplot may be empty)
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    plt.tight_layout()
    fig.legend(
        handles,
        labels,
        title='Method - '+ r'$\mathit{Itp}$ Algorithm',
        loc='upper center',
        title_fontsize=legend_fs + 6,
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        columnspacing=1.0,
        fontsize=legend_fs+4,
        frameon=False
    )

    os.makedirs(plots_dir, exist_ok=True)
    plot_name = os.path.join(plots_dir, f'{plot_case}_{layout}_{name}.pdf')
    # bbox_inches='tight' grows the canvas to fit the legend, however many rows it has
    plt.savefig(f"{plot_name}", format='pdf', bbox_inches='tight')
    # plt.show()
    print(f"Saved runtime comparison plot to {plot_name}")
