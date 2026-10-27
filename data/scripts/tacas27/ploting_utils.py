import re

import numpy as np
import pandas as pd

def rename_method_algorithm(method, algorithm):
    #     f"{method} - {rename_alg(algorithm)}"
    if algorithm.startswith(' '):
        algorithm = algorithm[1:]
    alg_mapping = {
        'weaker': r'$\mathtt{weaker}$',
        'weak': r'$\mathtt{weak}$',
        'mid': r'$\mathtt{mid}$',
        'strong': r'$\mathtt{strong}$',
        'stronger': r'$\mathtt{stronger}$',
        'e=1': r'$\epsilon=100\%$',
        'e=0.05': r'$\epsilon=5\%$',
        'e=0.005': r'$\epsilon=0.5\%$',
    }
    try:
        alg_str = alg_mapping[algorithm]
    except KeyError:
        alg_str = algorithm
    ret_str = method + ' - ' + alg_mapping.get(algorithm, algorithm)
    return ret_str

# def rename_alg(algorithm):
#     print(algorithm)
#     if algorithm.startswith(' '):
#         algorithm = algorithm[1:]
#     alg_mapping = {
#         'weaker': r'$\mathtt{weaker}$',
#         'weak': r'$\mathtt{weak}$',
#         'mid': r'$\mathtt{mid}$',
#         'strong': r'$\mathtt{strong}$',
#         'stronger': r'$\mathtt{stronger}$',
#         'e=1': r'$\epsilon=100\%$',
#         'e=0.05': r'$\epsilon=5\%$',
#         'e=0.005': r'$\epsilon=0.5\%$',
#     }
#     try:
#         ret_val = alg_mapping[algorithm]
#     except KeyError:
#         ret_val = algorithm
#     print(ret_val)
#     return ret_val

def rename_methods(df):
    method_mapping = {
        'naive': 'Direct',
        'fix-all': 'Fix-all',
        'prefer-all': 'Guide-all',
        'guide-all': 'Guide-all',
        'fix-guided': 'Hybrid',
        'hybrid': 'Hybrid',
        'verix incomplete random': 'VeriX - IR',
        'verix complete sensitivity': 'VeriX - CS'
    }
    df['Method'] = df['Method'].map(method_mapping)
    return df


def rename_single_methods(m):
    m = m.lower()
    method_mapping = {
        'naive': 'Direct',
        'fix-all': 'Fix-all',
        'prefer-all': 'Guide-all',
        'guide-all': 'Guide-all',
        'fix-guided': 'Hybrid',
        'hybrid': 'Hybrid',
        'verix incomplete random': 'VeriX - IR',
        'verix complete sensitivity': 'VeriX - CS'
    }

    return method_mapping[m]


def rename_itp(df):
    exp_itp_keys = {
        'itp_aweaker_bstrong': r'weaker',
        'itp_aweak_bstrong': r'weak',
        'itp_afactor_0.5_bstrong': r'mid',
        'itp_astrong_bstrong': r'strong',
        'itp_astronger_bstrong': r'stronger',
    }
    # newer stats prefix the itp name with the method, e.g. 'fix-all/itp_aweak_bstrong'
    df['Experiment Name'] = df['Experiment Name'].str.split('/').str[-1].map(exp_itp_keys)
    order = ['weaker', 'weak', 'mid', 'strong', 'stronger']
    df['Experiment Name'] = pd.Categorical(df['Experiment Name'], categories=order, ordered=True)
    df = df.sort_values(by=['Experiment Name', 'Model Size'])
    return df

plot_colors = {'Naive': 'blue', 'Direct': 'blue', 'Fix-all': 'red', 'Guide-all': 'green', 'Hybrid': 'orange', rename_single_methods('VeriX Incomplete Random'): 'red', rename_single_methods( 'VeriX Complete Sensitivity'): 'red'}

def rename_task_name(task):
    task = task.lower()
    task = task.replace(' ', '_')
    task_mapping = {
        'cifar_200': 'CIFAR10 ',
        'cifar': 'CIFAR10',
        'gtsrb_200': 'GTSRB',
        'gtsrb': 'GTSRB',
        'mnist_200': 'MNIST',
        'mnist': 'MNIST',
        'heart_attack_200': 'Heart attack risk',
        'heart_attack': 'Heart attack risk',
        'obesity_200': 'Obesity',
        'obesity': 'Obesity',
        'loan_200': 'Loan',
        'loan': 'Loan',
    }
    return task_mapping.get(task, task)


# Plot titles of the CNN model families (cnn_bench): narrow (N*) or wide, and how the feature maps are downsampled
cnn_family_titles = {
    'S': 'Stride-2 downsampling (S)',
    'AB': 'Average pooling (AB)',
    'NS': 'Narrow, stride-2 downsampling (NS)',
    'NAB': 'Narrow, average pooling (NAB)',
}


def cnn_family_title(family):
    return cnn_family_titles.get(family, family)


# Only these model depths (number of hidden layers) of the FC model families are plotted
tabular_depths = [1, 2, 3, 4, 6, 8]
image_depths = [1, 2, 3, 4, 6, 8, 10, 12, 14]
_depths_by_task = {'heart_attack': tabular_depths, 'obesity': tabular_depths, 'loan': tabular_depths,
                   'mnist': image_depths, 'cifar': image_depths, 'gtsrb': image_depths}


def shown_depths(task):
    # Accepts task names with a family suffix too, e.g. 'cifar_200'
    task = task.lower().replace(' ', '_').removesuffix('_200')
    return _depths_by_task[task]


# Timeout per sample [s] of each task (use_stats_timeouts replaces them with those of the stats)
timeouts = {'heart_attack': 300, 'obesity': 300, 'loan': 300, 'mnist': 600, 'cifar': 600, 'gtsrb': 840}


def draw_timeout_lines(ax, task):
    # Horizontal reference line of a PAR-2 plot at the timeout per sample, labeled at the right end. Its color is
    # not used by any method (plot_colors) or epsilon (epsilon_colors), and it is drawn over the data to stay visible.
    # Call it after setting the y-limits: a timeout above the shown range is not drawn.
    timeout = timeouts[task.lower().replace(' ', '_').removesuffix('_200')]
    if timeout > ax.get_ylim()[1]:
        return
    ax.axhline(timeout, color='teal', linestyle='--', linewidth=1.5, zorder=3)
    ax.text(0.99, timeout, f'timeout ({timeout:g} s)', transform=ax.get_yaxis_transform(), ha='right', va='bottom',
            color='teal', fontsize=tick_fs - 2, zorder=4, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))



def draw_model_names(ax, model_x):
    # Model names once, on a top x-axis at each model's #ReLUs (the bottom axis keeps the numeric scale).
    # model_x: (name, #ReLUs) pairs
    model_x = sorted(model_x, key=lambda m: m[1])
    top = ax.secondary_xaxis('top')
    top.set_xticks([x for _, x in model_x])
    top.set_xticklabels([m for m, _ in model_x], fontsize=tick_fs - 2, rotation=45, ha='left', rotation_mode='anchor')


# Common line format of all plots. Our methods are solid lines, colored by method (plot_colors); VeriX+ lines are
# dashed and colored by epsilon. Markers are hollow and tell apart the Itp algorithms when a plot shows several of
# them, otherwise the methods; for VeriX+ they tell apart the epsilons.
line_width = 1.5
marker_size = 8
method_markers = {'Direct': 'd', 'Guide-all': '^', 'Fix-all': 'v', 'Hybrid': 'x'}
itp_markers = {'weaker': 'd', 'weak': '^', 'mid': 'v', 'strong': 'x', 'stronger': 'o'}
epsilon_colors = {'100%': 'purple', '5%': 'saddlebrown', '0.5%': 'gray'}
epsilon_markers = {'100%': 'o', '5%': 'd', '0.5%': '^'}


def our_line_style(method, algorithm, several_itps):
    marker = itp_markers.get(algorithm, 'o') if several_itps else method_markers.get(method, 'o')
    return dict(color=plot_colors.get(method, 'black'), marker=marker, markerfacecolor='none',
                markersize=marker_size, linestyle='-', linewidth=line_width)


def verix_line_style(epsilon, linestyle='--'):
    return dict(color=epsilon_colors.get(epsilon, 'black'), marker=epsilon_markers.get(epsilon, 'X'),
                markerfacecolor='none', markersize=marker_size, linestyle=linestyle, linewidth=line_width)


# Font sizes
label_fs = 20
title_fs = 20
tick_fs = 14
legend_fs = 14


def read_stats(stats_file, task):
    # Parses the stats file of one task (of `runme.sh collect-stats`, all model widths) into a DataFrame with the
    # columns of the CSV tables of plot_fc_stats.py, and the set of the timeouts per sample [s] it states
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
    with open(stats_file, 'r') as file:
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

    return df, task_timeouts


def use_stats_timeouts(task, task_timeouts):
    # The timeouts of the modes differ (see modes.sh), so use those of the stats (of read_stats) instead of the
    # defaults in timeouts; draw_timeout_lines and the y-limits of the plotting scripts read the same dictionary
    if task_timeouts:
        if len(task_timeouts) > 1:
            print(f'WARNING: {task} has several timeouts per sample: {sorted(task_timeouts)}; using the largest one')
        timeouts[task] = max(task_timeouts)
