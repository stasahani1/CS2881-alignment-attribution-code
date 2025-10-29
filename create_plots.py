#!/usr/bin/env python3
"""
Create 3 plots for Figure 2a:
1. no_inst_ASR_multiple_nosys vs averaged zero-shot accuracy
2. ASR_gcg vs averaged zero-shot accuracy
3. inst_ASR_multiple_nosys vs averaged zero-shot accuracy
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Data extracted from results/fig2a/
data = {
    'Wanda': {
        'zero_shot': 0.3550,
        'inst_ASR_multiple_nosys': 0.6720,
        'ASR_gcg': 0.2900,
        'no_inst_ASR_multiple_nosys': 0.9020,
    },
    'SNIP': {
        'zero_shot': 0.3367,
        'inst_ASR_multiple_nosys': 0.9220,
        'ASR_gcg': 0.9800,
        'no_inst_ASR_multiple_nosys': 0.9640,
    },
    'Set Difference': {
        'zero_shot': 0.5825,
        'inst_ASR_multiple_nosys': 0.0000,
        'ASR_gcg': 0.0900,
        'no_inst_ASR_multiple_nosys': 0.2740,
    },
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
                'inst_ASR_multiple_nosys': metrics.get('inst_ASR_multiple_nosys', None),
                'ASR_gcg': metrics.get('ASR_gcg', None),
                'no_inst_ASR_multiple_nosys': metrics.get('no_inst_ASR_multiple_nosys', None),
                'sparsity': os.path.basename(sparsity_dir)
            }
            setdiff_data.append(data_point)

    return setdiff_data

setdiff_data = read_setdiff_data()

# Colors for each method
colors = {
    'Wanda': '#1f77b4',  # blue
    'SNIP': '#ff7f0e',   # orange
    'Set Difference': '#2ca02c',  # green
}

# Marker styles
markers = {
    'Wanda': 'o',
    'SNIP': 's',
    'Set Difference': '^',
}

# Create the 3 plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

asr_metrics = [
    ('no_inst_ASR_multiple_nosys', 'ASR (no_inst, multiple samples, no sys)'),
    ('ASR_gcg', 'ASR (GCG Attack)'),
    ('inst_ASR_multiple_nosys', 'ASR (inst, multiple samples, no sys)'),
]

for idx, (metric_key, metric_label) in enumerate(asr_metrics):
    ax = axes[idx]

    # Plot each method
    for method in ['Wanda', 'SNIP', 'Set Difference']:
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

        # Add text label
        ax.annotate(method,
                   (x, y),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   alpha=0.7)

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
    for method in ['Wanda', 'SNIP', 'Set Difference']:
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

        # Add text label
        ax.annotate(method,
                   (x, y),
                   xytext=(8, 8),
                   textcoords='offset points',
                   fontsize=11,
                   alpha=0.7)

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
print(f"{'Method':<20} {'Zero-shot':>12} {'inst_ASR_mult':>15} {'ASR_gcg':>12} {'no_inst_ASR':>15}")
print("-"*80)
for method in ['Wanda', 'SNIP', 'Set Difference']:
    print(f"{method:<20} {data[method]['zero_shot']:>12.4f} "
          f"{data[method]['inst_ASR_multiple_nosys']:>15.4f} "
          f"{data[method]['ASR_gcg']:>12.4f} "
          f"{data[method]['no_inst_ASR_multiple_nosys']:>15.4f}")
print("-"*80)
print("\nSet Difference (all sparsity levels):")
print("-"*80)
for point in setdiff_data:
    print(f"{point['sparsity']:<20} {point['zero_shot']:>12.4f} "
          f"{point['inst_ASR_multiple_nosys']:>15.4f} "
          f"{point['ASR_gcg']:>12.4f} "
          f"{point['no_inst_ASR_multiple_nosys']:>15.4f}")
print("="*80)
