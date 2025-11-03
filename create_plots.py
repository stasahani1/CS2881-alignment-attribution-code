#!/usr/bin/env python3
"""
Create 3 plots for Figure 2a:
1. ASR_vanilla (inst_ASR_basic) vs averaged zero-shot accuracy
2. ASR_adv_suffix (ASR_gcg) vs averaged zero-shot accuracy
3. ASR_adv_decoding (no_inst_ASR_multiple_nosys) vs averaged zero-shot accuracy
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Paths to log files
LOG_PATHS = {
    'Original': 'results/fig2a/original/log_original.txt',
    'Wanda': 'results/fig2a/wanda_0.01/log_wanda.txt',
    'SNIP': 'results/fig2a/snip/log_wandg.txt',
    'Set Difference': 'results/fig2a/snip_setdiff/log_wandg_set_difference.txt',
}

# Read data from log files
def read_log_file(filepath, method_name):
    """Read metrics from a log file and return a dictionary of metric: score"""
    metrics = {}

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return metrics

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                # Handle both formats: with and without p,q columns
                if 'set_difference' in filepath:
                    # Format: method, actual_sparsity, p, q, metric, score
                    if len(parts) >= 6:
                        metric_name = parts[4]
                        score = float(parts[5])
                        metrics[metric_name] = score
                else:
                    # Format: method, actual_sparsity, metric, score
                    metric_name = parts[2]
                    score = float(parts[3])
                    metrics[metric_name] = score

    return metrics

# Load data from log files
data = {}
for method, filepath in LOG_PATHS.items():
    metrics = read_log_file(filepath, method)
    data[method] = {
        'zero_shot': metrics.get('averaged', 0.0),
        'inst_ASR_basic': metrics.get('inst_ASR_basic', 0.0),
        'ASR_gcg': metrics.get('ASR_gcg', 0.0),
        'no_inst_ASR_multiple_nosys': metrics.get('no_inst_ASR_multiple_nosys', 0.0),
    }

# Read set difference data from all sparsity folders
def read_setdiff_data():
    """Read data from results/fig2a/snip_setdiff/sparsity_*/log_wandg_set_difference.txt"""
    setdiff_data = []

    # Find all sparsity directories
    sparsity_dirs = sorted(glob.glob('results/fig2a/snip_setdiff/sparsity_*'))

    for sparsity_dir in sparsity_dirs:
        log_file = os.path.join(sparsity_dir, 'log_wandg_set_difference.txt')
        if not os.path.exists(log_file):
            continue

        # Parse the log file
        metrics = {}
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    metric_name = parts[4]
                    score = float(parts[5])
                    metrics[metric_name] = score

        # Extract the relevant metrics
        if 'averaged' in metrics:
            data_point = {
                'zero_shot': metrics.get('averaged', None),
                'inst_ASR_basic': metrics.get('inst_ASR_basic', None),
                'ASR_gcg': metrics.get('ASR_gcg', None),
                'no_inst_ASR_multiple_nosys': metrics.get('no_inst_ASR_multiple_nosys', None),
                'sparsity': os.path.basename(sparsity_dir)
            }
            setdiff_data.append(data_point)

    return setdiff_data

setdiff_data = read_setdiff_data()

# Colors for each method
colors = {
    'Original': '#9467bd',  # purple
    'Wanda': '#1f77b4',  # blue
    'SNIP': '#ff7f0e',   # orange
    'Set Difference': '#2ca02c',  # green
}

# Marker styles
markers = {
    'Original': 'D',  # diamond
    'Wanda': 'o',
    'SNIP': 's',
    'Set Difference': '^',
}

# Create the 3 plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

asr_metrics = [
    ('inst_ASR_basic', 'ASR_vanilla'),
    ('ASR_gcg', 'ASR_adv_suffix'),
    ('no_inst_ASR_multiple_nosys', 'ASR_adv_decoding'),
]

for idx, (metric_key, metric_label) in enumerate(asr_metrics):
    ax = axes[idx]

    # Plot each method
    for method in ['Original', 'Wanda', 'SNIP', 'Set Difference']:
        x = data[method]['zero_shot']
        y = data[method][metric_key]

        ax.scatter(x, y,
                  color=colors[method],
                  marker=markers[method],
                  s=200,
                  label=method,
                  edgecolors='black',
                  linewidth=1.5,
                  alpha=0.8)

    # Plot all set difference sparsity points (green triangles)
    for point in setdiff_data:
        x = point['zero_shot']
        y = point[metric_key]
        if x is not None and y is not None:
            ax.scatter(x, y,
                      color=colors['Set Difference'],
                      marker=markers['Set Difference'],
                      s=200,
                      edgecolors='black',
                      linewidth=1.5,
                      alpha=0.8)

    # Formatting
    ax.set_xlabel('Zero-shot Accuracy (averaged)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 2a: {metric_label}\nvs Zero-shot Accuracy',
                fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Set axis limits with some padding
    ax.set_xlim(0.3, 0.65)
    ax.set_ylim(-0.05, 1.05)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('results/fig2a/figure_2a_plots.png', dpi=300, bbox_inches='tight')
print("Saved: results/fig2a/figure_2a_plots.png")

# Also create individual plots
for idx, (metric_key, metric_label) in enumerate(asr_metrics):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each method
    for method in ['Original', 'Wanda', 'SNIP', 'Set Difference']:
        x = data[method]['zero_shot']
        y = data[method][metric_key]

        ax.scatter(x, y,
                  color=colors[method],
                  marker=markers[method],
                  s=300,
                  label=method,
                  edgecolors='black',
                  linewidth=2,
                  alpha=0.8)

    # Plot all set difference sparsity points (green triangles)
    for point in setdiff_data:
        x = point['zero_shot']
        y = point[metric_key]
        if x is not None and y is not None:
            ax.scatter(x, y,
                      color=colors['Set Difference'],
                      marker=markers['Set Difference'],
                      s=300,
                      edgecolors='black',
                      linewidth=2,
                      alpha=0.8)

    # Formatting
    ax.set_xlabel('Zero-shot Accuracy (averaged)', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
    ax.set_title(f'Figure 2a: {metric_label}\nvs Zero-shot Accuracy',
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)

    # Set axis limits with some padding
    ax.set_xlim(0.3, 0.65)
    ax.set_ylim(-0.05, 1.05)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    filename = f'results/fig2a/figure_2a_{metric_key}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

# Print the data table
print("\n" + "="*80)
print("DATA SUMMARY")
print("="*80)
print(f"{'Method':<20} {'Zero-shot':>12} {'ASR_vanilla':>15} {'ASR_adv_suffix':>15} {'ASR_adv_dec':>15}")
print("-"*80)
for method in ['Original', 'Wanda', 'SNIP', 'Set Difference']:
    print(f"{method:<20} {data[method]['zero_shot']:>12.4f} "
          f"{data[method]['inst_ASR_basic']:>15.4f} "
          f"{data[method]['ASR_gcg']:>15.4f} "
          f"{data[method]['no_inst_ASR_multiple_nosys']:>15.4f}")
print("-"*80)
print("\nSet Difference (all sparsity levels):")
print("-"*80)
for point in setdiff_data:
    print(f"{point['sparsity']:<20} {point['zero_shot']:>12.4f} "
          f"{point['inst_ASR_basic']:>15.4f} "
          f"{point['ASR_gcg']:>15.4f} "
          f"{point['no_inst_ASR_multiple_nosys']:>15.4f}")
print("="*80)
