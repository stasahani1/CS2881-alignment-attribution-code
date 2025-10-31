"""
Analyze neuron drift patterns across training.

This script:
1. Loads all drift logs from fine-tuning
2. Computes statistics and trends
3. Performs statistical tests (Mann-Whitney U)
4. Generates visualizations
5. Saves results and figures
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_drift_logs(log_dir: str) -> Dict[int, Dict]:
    """
    Load all drift logs from directory.

    Args:
        log_dir: Directory containing drift log JSON files

    Returns:
        Dictionary mapping step number to drift metrics
    """
    drift_logs = {}
    log_files = sorted(Path(log_dir).glob("drift_step_*.json"))

    print(f"Loading drift logs from {log_dir}")
    print(f"Found {len(log_files)} log files")

    for log_file in log_files:
        with open(log_file, "r") as f:
            data = json.load(f)
            step = data["step"]
            drift_logs[step] = data["metrics"]

    print(f"Loaded drift data for {len(drift_logs)} checkpoints")
    return drift_logs


def extract_time_series(
    drift_logs: Dict[int, Dict],
    metric_name: str,
    stat_name: str = "mean"
) -> Dict[str, List]:
    """
    Extract time series for a specific metric and statistic.

    Args:
        drift_logs: Drift logs by step
        metric_name: Name of metric (e.g., "cosine_similarity")
        stat_name: Statistic to extract (e.g., "mean", "std")

    Returns:
        Dictionary mapping group name to list of values over time
    """
    steps = sorted(drift_logs.keys())
    time_series = {}

    # Get group names from first log
    first_step = steps[0]
    group_names = drift_logs[first_step][metric_name].keys()

    # Extract time series for each group
    for group_name in group_names:
        values = []
        for step in steps:
            value = drift_logs[step][metric_name][group_name][stat_name]
            values.append(value)
        time_series[group_name] = values

    return time_series, steps


def plot_time_series(
    drift_logs: Dict[int, Dict],
    metric_name: str,
    output_dir: str,
    stat_name: str = "mean"
):
    """
    Plot metric over time for all groups.

    Args:
        drift_logs: Drift logs by step
        metric_name: Name of metric to plot
        output_dir: Output directory for plots
        stat_name: Statistic to plot
    """
    time_series, steps = extract_time_series(drift_logs, metric_name, stat_name)

    plt.figure(figsize=(10, 6))

    for group_name, values in time_series.items():
        plt.plot(steps, values, marker='o', label=group_name, linewidth=2)

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel(f"{metric_name.replace('_', ' ').title()} ({stat_name})", fontsize=12)
    plt.title(f"{metric_name.replace('_', ' ').title()} Over Training", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{metric_name}_{stat_name}_over_time.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_final_distributions(
    drift_logs: Dict[int, Dict],
    metric_name: str,
    output_dir: str
):
    """
    Plot box plots of final metric distributions.

    Args:
        drift_logs: Drift logs by step
        metric_name: Name of metric to plot
        output_dir: Output directory for plots
    """
    # Get final step
    final_step = max(drift_logs.keys())
    final_metrics = drift_logs[final_step][metric_name]

    # Prepare data for box plot
    group_names = list(final_metrics.keys())
    data = []
    positions = []

    for i, group_name in enumerate(group_names):
        stats_dict = final_metrics[group_name]
        # Approximate distribution from summary statistics
        # Using mean, std, and percentiles
        mean = stats_dict["mean"]
        std = stats_dict["std"]
        q25 = stats_dict["q25"]
        median = stats_dict["median"]
        q75 = stats_dict["q75"]

        # Create box plot data
        data.append({
            'label': group_name,
            'mean': mean,
            'median': median,
            'q1': q25,
            'q3': q75,
            'whislo': stats_dict["min"],
            'whishi': stats_dict["max"],
        })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot boxes manually
    for i, d in enumerate(data):
        # Draw box
        box = plt.Rectangle(
            (i + 0.6, d['q1']),
            0.8,
            d['q3'] - d['q1'],
            fill=True,
            alpha=0.5,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)

        # Draw median line
        ax.plot([i + 0.6, i + 1.4], [d['median'], d['median']],
                color='red', linewidth=2)

        # Draw whiskers
        ax.plot([i + 1, i + 1], [d['q1'], d['whislo']],
                color='black', linestyle='--', linewidth=1)
        ax.plot([i + 1, i + 1], [d['q3'], d['whishi']],
                color='black', linestyle='--', linewidth=1)

        # Draw mean as a point
        ax.scatter([i + 1], [d['mean']], color='blue', s=100,
                   marker='D', zorder=3, label='Mean' if i == 0 else '')

    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([d['label'] for d in data], rotation=15, ha='right')
    ax.set_ylabel(f"{metric_name.replace('_', ' ').title()}", fontsize=12)
    ax.set_title(f"Final {metric_name.replace('_', ' ').title()} Distribution", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{metric_name}_final_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def statistical_tests(
    drift_logs: Dict[int, Dict],
    metric_name: str
) -> Dict:
    """
    Perform statistical tests comparing groups.

    Uses final checkpoint for comparison.

    Args:
        drift_logs: Drift logs by step
        metric_name: Name of metric to test

    Returns:
        Dictionary with test results
    """
    # Get final step
    final_step = max(drift_logs.keys())
    final_metrics = drift_logs[final_step][metric_name]

    results = {}

    # Compare each safety group vs random
    group_names = list(final_metrics.keys())

    # Find random group
    random_group = None
    for name in group_names:
        if "random" in name.lower():
            random_group = name
            break

    if random_group is None:
        print("Warning: No random group found for statistical testing")
        return results

    random_stats = final_metrics[random_group]

    print(f"\n{metric_name.upper()} - Statistical Comparisons vs {random_group}:")
    print("=" * 60)

    for group_name in group_names:
        if group_name == random_group:
            continue

        group_stats = final_metrics[group_name]

        # We don't have individual data points, so we'll compare means
        # and report effect size (Cohen's d)
        mean_diff = group_stats["mean"] - random_stats["mean"]

        # Cohen's d: difference in means / pooled standard deviation
        pooled_std = np.sqrt(
            (group_stats["std"]**2 + random_stats["std"]**2) / 2
        )

        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
        else:
            cohens_d = 0.0

        results[f"{group_name}_vs_{random_group}"] = {
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
            "group_mean": group_stats["mean"],
            "random_mean": random_stats["mean"],
            "group_std": group_stats["std"],
            "random_std": random_stats["std"],
        }

        print(f"{group_name}:")
        print(f"  Mean: {group_stats['mean']:.4f} (random: {random_stats['mean']:.4f})")
        print(f"  Difference: {mean_diff:+.4f}")
        print(f"  Cohen's d: {cohens_d:+.4f} ", end="")

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            print("(negligible)")
        elif abs(cohens_d) < 0.5:
            print("(small)")
        elif abs(cohens_d) < 0.8:
            print("(medium)")
        else:
            print("(large)")

    return results


def generate_summary_report(
    drift_logs: Dict[int, Dict],
    statistical_results: Dict,
    output_dir: str
):
    """
    Generate a text summary report.

    Args:
        drift_logs: Drift logs by step
        statistical_results: Results from statistical tests
        output_dir: Output directory
    """
    output_path = os.path.join(output_dir, "drift_analysis_report.txt")

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NEURON DRIFT ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Training summary
        steps = sorted(drift_logs.keys())
        f.write(f"Training Steps Analyzed: {len(steps)}\n")
        f.write(f"Step Range: {steps[0]} - {steps[-1]}\n\n")

        # Final metrics
        final_step = max(drift_logs.keys())
        final_metrics = drift_logs[final_step]

        for metric_name in ["cosine_similarity", "l2_distance", "relative_change"]:
            if metric_name not in final_metrics:
                continue

            f.write("-" * 70 + "\n")
            f.write(f"{metric_name.upper().replace('_', ' ')}\n")
            f.write("-" * 70 + "\n\n")

            f.write("Final Values by Group:\n")
            for group_name, stats in final_metrics[metric_name].items():
                f.write(f"  {group_name}:\n")
                f.write(f"    Mean:   {stats['mean']:.6f}\n")
                f.write(f"    Std:    {stats['std']:.6f}\n")
                f.write(f"    Median: {stats['median']:.6f}\n")
                f.write(f"    Range:  [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"    Count:  {stats['count']}\n")

            # Statistical comparisons
            if metric_name in statistical_results:
                f.write("\nStatistical Comparisons:\n")
                for comparison, results in statistical_results[metric_name].items():
                    f.write(f"  {comparison}:\n")
                    f.write(f"    Mean Difference: {results['mean_diff']:+.6f}\n")
                    f.write(f"    Cohen's d:       {results['cohens_d']:+.6f}\n")

            f.write("\n")

        # Hypothesis evaluation
        f.write("=" * 70 + "\n")
        f.write("HYPOTHESIS EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        # Check if safety neurons drift more (H1) or similar (H2)
        safety_groups = [k for k in final_metrics["cosine_similarity"].keys()
                        if "safety" in k.lower()]
        random_group = [k for k in final_metrics["cosine_similarity"].keys()
                       if "random" in k.lower()]

        if safety_groups and random_group:
            random_cos_sim = final_metrics["cosine_similarity"][random_group[0]]["mean"]

            f.write("H1 (Fragile Safety): Safety neurons drift MORE than random\n")
            f.write("  → Lower cosine similarity, higher L2 distance\n\n")

            f.write("H2 (Pathway Creation): Safety neurons drift LESS or SIMILAR\n")
            f.write("  → Supports alternative pathway hypothesis\n\n")

            f.write("Results:\n")
            for group in safety_groups:
                safety_cos_sim = final_metrics["cosine_similarity"][group]["mean"]
                diff = safety_cos_sim - random_cos_sim

                f.write(f"  {group}:\n")
                f.write(f"    Cosine similarity: {safety_cos_sim:.6f} "
                       f"(random: {random_cos_sim:.6f})\n")
                f.write(f"    Difference: {diff:+.6f}\n")

                if diff < -0.01:  # Safety drifts more
                    f.write("    → Supports H1 (Fragile Safety)\n")
                elif abs(diff) < 0.01:  # Similar drift
                    f.write("    → Supports H2 (Pathway Creation)\n")
                else:  # Safety drifts less
                    f.write("    → Supports H2 (Pathway Creation, strong)\n")
                f.write("\n")

    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze neuron drift patterns"
    )
    parser.add_argument(
        "--drift_log_dir",
        type=str,
        default="/dev/shm/drift_logs",
        help="Directory containing drift logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/CS2881-alignment-attribution-code/results",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("Analyzing Neuron Drift Patterns")
    print("=" * 70)
    print(f"Drift logs: {args.drift_log_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Load drift logs
    drift_logs = load_drift_logs(args.drift_log_dir)

    if not drift_logs:
        print("Error: No drift logs found!")
        return

    # Generate time series plots
    print("\nGenerating time series plots...")
    for metric_name in ["cosine_similarity", "l2_distance", "relative_change"]:
        plot_time_series(drift_logs, metric_name, figures_dir, stat_name="mean")
        plot_time_series(drift_logs, metric_name, figures_dir, stat_name="std")

    # Generate distribution plots
    print("\nGenerating distribution plots...")
    for metric_name in ["cosine_similarity", "l2_distance", "relative_change"]:
        plot_final_distributions(drift_logs, metric_name, figures_dir)

    # Statistical tests
    print("\nPerforming statistical tests...")
    statistical_results = {}
    for metric_name in ["cosine_similarity", "l2_distance", "relative_change"]:
        results = statistical_tests(drift_logs, metric_name)
        statistical_results[metric_name] = results

    # Save statistical results
    stats_path = os.path.join(args.output_dir, "statistical_results.json")
    with open(stats_path, "w") as f:
        json.dump(statistical_results, f, indent=2)
    print(f"\nSaved statistical results: {stats_path}")

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(drift_logs, statistical_results, args.output_dir)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"Results saved to: {args.output_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
