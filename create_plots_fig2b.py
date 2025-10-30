#!/usr/bin/env python3
"""
Create 3 plots for Figure 2b (rank-based methods):
1. no_inst_ASR_multiple_nosys vs averaged zero-shot accuracy
2. ASR_gcg vs averaged zero-shot accuracy
3. inst_ASR_multiple_nosys vs averaged zero-shot accuracy

Compares:
- ActSVD (top rank removal, rank=1)
- Orthogonal Projection (different rank combinations)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def parse_log_file(log_path, rank_label):
    """Parse a log file and extract metrics"""
    metrics = {'rank': rank_label}

    with open(log_path, 'r') as f:
        lines = f.readlines()

        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue

            metric_name = parts[2]
            score_str = parts[3]

            # Handle the inst_ prefix issue
            if len(parts) >= 4 and parts[1] in ['inst_', 'no_inst_']:
                metric_name = parts[2]
                score_str = parts[3]

            try:
                score = float(score_str)
                metrics[metric_name] = score
            except ValueError:
                continue

    return metrics

def read_actsvd_data():
    """Read ActSVD top rank removal data"""
    log_file = 'results/fig2b/actSVD_top/log_low_rank.txt'

    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found")
        return None

    data = {}
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                metric_name = parts[2]
                score = float(parts[3])
                data[metric_name] = score

    return data

def read_orth_proj_data():
    """Read all orthogonal projection data"""
    orth_proj_data = []

    # Find all rank combination directories
    rank_dirs = sorted(glob.glob('results/fig2b/orth_proj/ru_*_rs_*'))

    for rank_dir in rank_dirs:
        log_file = os.path.join(rank_dir, 'log.txt')
        if not os.path.exists(log_file):
            continue

        # Extract ru and rs from directory name
        dir_name = os.path.basename(rank_dir)
        parts = dir_name.split('_')
        ru = int(parts[1])
        rs = int(parts[3])
        orth_dim = min(ru, 4096 - rs)

        # Parse the log file
        metrics = parse_log_file(log_file, f"ru={ru}, rs={rs}")
        metrics['ru'] = ru
        metrics['rs'] = rs
        metrics['orth_dim'] = orth_dim

        if 'averaged' in metrics:
            orth_proj_data.append(metrics)

    return orth_proj_data

# Read all data
actsvd_data = read_actsvd_data()
orth_proj_data = read_orth_proj_data()

# Print data summary
print("\n" + "="*100)
print("DATA SUMMARY - Figure 2b")
print("="*100)

if actsvd_data:
    print("\nActSVD (rank=1 removal):")
    print(f"  Zero-shot: {actsvd_data.get('averaged', 'N/A'):.4f}")
    print(f"  inst_ASR_multiple_nosys: {actsvd_data.get('inst_ASR_multiple_nosys', 'N/A'):.4f}")
    print(f"  ASR_gcg: {actsvd_data.get('ASR_gcg', 'N/A'):.4f}")
    print(f"  no_inst_ASR_multiple_nosys: {actsvd_data.get('no_inst_ASR_multiple_nosys', 'N/A'):.4f}")

print(f"\nOrthogonal Projection ({len(orth_proj_data)} experiments):")
print(f"{'ru':<6} {'rs':<6} {'orth_dim':<10} {'Zero-shot':>12} {'inst_ASR':>12} {'ASR_gcg':>12} {'no_inst_ASR':>15}")
print("-"*100)
for data_point in orth_proj_data:
    print(f"{data_point['ru']:<6} {data_point['rs']:<6} {data_point['orth_dim']:<10} "
          f"{data_point.get('averaged', 0):.4f}      "
          f"{data_point.get('inst_ASR_multiple_nosys', 0):.4f}      "
          f"{data_point.get('ASR_gcg', 0):.4f}      "
          f"{data_point.get('no_inst_ASR_multiple_nosys', 0):.4f}")
print("="*100)

# Colors and markers
colors = {
    'ActSVD': '#1f77b4',  # blue
    'Orth_Proj': '#2ca02c',  # green
}

markers = {
    'ActSVD': 'o',
    'Orth_Proj': '^',
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

    # Plot ActSVD if available
    if actsvd_data and metric_key in actsvd_data:
        x = actsvd_data.get('averaged', 0)
        y = actsvd_data.get(metric_key, 0)

        ax.scatter(x, y,
                  color=colors['ActSVD'],
                  marker=markers['ActSVD'],
                  s=250,
                  label='ActSVD (rank=1)',
                  edgecolors='black',
                  linewidth=1.5,
                  alpha=0.8,
                  zorder=5)

        ax.annotate('ActSVD',
                   (x, y),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9,
                   alpha=0.7)

    # Plot all orthogonal projection points
    x_vals = []
    y_vals = []
    labels = []

    for point in orth_proj_data:
        if metric_key in point:
            x = point.get('averaged', 0)
            y = point.get(metric_key, 0)
            x_vals.append(x)
            y_vals.append(y)
            labels.append(f"orth_dim={point['orth_dim']}")

            ax.scatter(x, y,
                      color=colors['Orth_Proj'],
                      marker=markers['Orth_Proj'],
                      s=200,
                      edgecolors='black',
                      linewidth=1.5,
                      alpha=0.8)

    # Add one legend entry for orthogonal projection
    if x_vals:
        ax.scatter([], [],
                  color=colors['Orth_Proj'],
                  marker=markers['Orth_Proj'],
                  s=200,
                  label='Orthogonal Projection',
                  edgecolors='black',
                  linewidth=1.5,
                  alpha=0.8)

    # Formatting
    ax.set_xlabel('Zero-shot Accuracy (averaged)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 2b: {metric_label}\nvs Zero-shot Accuracy',
                fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Set axis limits with some padding
    if x_vals or actsvd_data:
        all_x = x_vals + ([actsvd_data.get('averaged', 0)] if actsvd_data else [])
        all_y = y_vals + ([actsvd_data.get(metric_key, 0)] if actsvd_data and metric_key in actsvd_data else [])

        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_range = x_max - x_min
            y_range = y_max - y_min

            ax.set_xlim(x_min - 0.1 * x_range if x_range > 0 else x_min - 0.05,
                       x_max + 0.1 * x_range if x_range > 0 else x_max + 0.05)
            ax.set_ylim(max(-0.05, y_min - 0.1 * y_range if y_range > 0 else y_min - 0.05),
                       min(1.05, y_max + 0.1 * y_range if y_range > 0 else y_max + 0.05))

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

plt.tight_layout()
output_file = 'results/fig2b/figure_2b_plots.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_file}")

# Also create individual plots
for idx, (metric_key, metric_label) in enumerate(asr_metrics):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ActSVD if available
    if actsvd_data and metric_key in actsvd_data:
        x = actsvd_data.get('averaged', 0)
        y = actsvd_data.get(metric_key, 0)

        ax.scatter(x, y,
                  color=colors['ActSVD'],
                  marker=markers['ActSVD'],
                  s=300,
                  label='ActSVD (rank=1)',
                  edgecolors='black',
                  linewidth=2,
                  alpha=0.8,
                  zorder=5)

        ax.annotate('ActSVD',
                   (x, y),
                   xytext=(8, 8),
                   textcoords='offset points',
                   fontsize=11,
                   alpha=0.7)

    # Plot all orthogonal projection points
    x_vals = []
    y_vals = []

    for point in orth_proj_data:
        if metric_key in point:
            x = point.get('averaged', 0)
            y = point.get(metric_key, 0)
            x_vals.append(x)
            y_vals.append(y)

            ax.scatter(x, y,
                      color=colors['Orth_Proj'],
                      marker=markers['Orth_Proj'],
                      s=300,
                      edgecolors='black',
                      linewidth=2,
                      alpha=0.8)

    # Add one legend entry for orthogonal projection
    if x_vals:
        ax.scatter([], [],
                  color=colors['Orth_Proj'],
                  marker=markers['Orth_Proj'],
                  s=300,
                  label='Orthogonal Projection',
                  edgecolors='black',
                  linewidth=2,
                  alpha=0.8)

    # Formatting
    ax.set_xlabel('Zero-shot Accuracy (averaged)', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
    ax.set_title(f'Figure 2b: {metric_label}\nvs Zero-shot Accuracy',
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)

    # Set axis limits with some padding
    if x_vals or actsvd_data:
        all_x = x_vals + ([actsvd_data.get('averaged', 0)] if actsvd_data else [])
        all_y = y_vals + ([actsvd_data.get(metric_key, 0)] if actsvd_data and metric_key in actsvd_data else [])

        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_range = x_max - x_min
            y_range = y_max - y_min

            ax.set_xlim(x_min - 0.1 * x_range if x_range > 0 else x_min - 0.05,
                       x_max + 0.1 * x_range if x_range > 0 else x_max + 0.05)
            ax.set_ylim(max(-0.05, y_min - 0.1 * y_range if y_range > 0 else y_min - 0.05),
                       min(1.05, y_max + 0.1 * y_range if y_range > 0 else y_max + 0.05))

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    filename = f'results/fig2b/figure_2b_{metric_key}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

print("\nAll plots generated successfully!")
