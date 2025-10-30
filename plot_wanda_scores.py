"""
Simple plotting utility for Wanda score analysis.

Visualizes how Wanda scores change during fine-tuning, particularly for frozen safety-critical neurons.
Used to interpret results from Experiment 1 (Frozen-Regime Fine-Tuning).
"""

import argparse
import os
import torch

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Required plotting libraries not available: {e}")
    print("Please ensure matplotlib and numpy are installed:")
    print("  pip install matplotlib numpy")
    exit(1)


def load_scores(scores_path):
    """
    Load Wanda scores from file.
    Handles both sparse format (dict with indices/scores) and legacy full scores.
    """
    scores = torch.load(scores_path)
    return scores


def extract_safety_neuron_scores(scores_dict, masks_dict):
    """
    Extract scores for safety-critical neurons only.

    Args:
        scores_dict: Dictionary of scores per layer
        masks_dict: Dictionary of boolean masks indicating safety neurons

    Returns:
        Tensor of scores for safety-critical neurons
    """
    safety_scores = []

    for layer_name in masks_dict.keys():
        if layer_name not in scores_dict:
            continue

        score_data = scores_dict[layer_name]
        mask = masks_dict[layer_name].flatten()

        # Handle sparse storage format
        if isinstance(score_data, dict):
            # Sparse format: only top-k neurons stored
            # For safety neurons, these are the ones we care about
            safety_scores.append(score_data['scores'])
        else:
            # Legacy format: full scores
            safety_scores.append(score_data.flatten()[mask])

    if len(safety_scores) > 0:
        return torch.cat(safety_scores)
    return torch.tensor([])


def plot_score_distributions(original_scores, fine_tuned_scores, masks, output_dir):
    """
    Plot histograms of score distributions before and after fine-tuning.
    """
    # Extract safety neuron scores
    orig_safety = extract_safety_neuron_scores(original_scores, masks)
    ft_safety = extract_safety_neuron_scores(fine_tuned_scores, masks)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Overlaid histograms
    axes[0].hist(orig_safety.numpy(), bins=50, alpha=0.5, label='Original', color='blue')
    axes[0].hist(ft_safety.numpy(), bins=50, alpha=0.5, label='Fine-tuned', color='red')
    axes[0].set_xlabel('Wanda Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Wanda Score Distributions\n(Safety-Critical Neurons)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Score changes (percentage)
    # Match original and fine-tuned scores
    if len(orig_safety) == len(ft_safety):
        pct_changes = ((ft_safety - orig_safety) / (orig_safety + 1e-10) * 100).numpy()
        axes[1].hist(pct_changes, bins=50, color='purple', alpha=0.7)
        axes[1].set_xlabel('Score Change (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Wanda Score Changes\n(Frozen Safety Neurons)')
        axes[1].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].grid(alpha=0.3)

        # Add statistics
        mean_change = pct_changes.mean()
        median_change = np.median(pct_changes)
        axes[1].text(0.05, 0.95, f'Mean: {mean_change:.1f}%\nMedian: {median_change:.1f}%',
                     transform=axes[1].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[1].text(0.5, 0.5, 'Score lengths mismatch\n(cannot compute changes)',
                     transform=axes[1].transAxes, ha='center', va='center')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'wanda_score_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_score_comparison_stats(original_scores, fine_tuned_scores, masks, output_dir):
    """
    Plot statistical comparison of scores.
    """
    # Extract scores
    orig_safety = extract_safety_neuron_scores(original_scores, masks)
    ft_safety = extract_safety_neuron_scores(fine_tuned_scores, masks)

    # Compute statistics
    stats = {
        'Original': {
            'Mean': orig_safety.mean().item(),
            'Median': orig_safety.median().item(),
            'Std': orig_safety.std().item(),
        },
        'Fine-tuned': {
            'Mean': ft_safety.mean().item(),
            'Median': ft_safety.median().item(),
            'Std': ft_safety.std().item(),
        }
    }

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ['Mean', 'Median', 'Std']
    x = np.arange(len(metrics))
    width = 0.35

    orig_values = [stats['Original'][m] for m in metrics]
    ft_values = [stats['Fine-tuned'][m] for m in metrics]

    ax.bar(x - width/2, orig_values, width, label='Original', color='blue', alpha=0.7)
    ax.bar(x + width/2, ft_values, width, label='Fine-tuned', color='red', alpha=0.7)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Wanda Score')
    ax.set_title('Statistical Comparison\n(Safety-Critical Neurons)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'wanda_score_statistics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

    # Print statistics
    print("\n" + "="*60)
    print("WANDA SCORE STATISTICS (Safety-Critical Neurons)")
    print("="*60)
    for regime in ['Original', 'Fine-tuned']:
        print(f"\n{regime}:")
        for metric, value in stats[regime].items():
            print(f"  {metric}: {value:.6f}")

    # Compute change
    if len(orig_safety) == len(ft_safety):
        mean_pct_change = ((ft_safety.mean() - orig_safety.mean()) / orig_safety.mean() * 100).item()
        print(f"\nMean Score Change: {mean_pct_change:+.2f}%")

        if mean_pct_change < -10:
            print("\n→ Supports Hypothesis A: Representational drift causing score drops")
            print("  (Frozen neurons become 'stranded' as model reorganizes)")
        elif abs(mean_pct_change) < 5:
            print("\n→ Supports Hypothesis B: Scores remain stable despite freezing")
            print("  (Global redistribution rather than local drift)")
        else:
            print("\n→ Unclear: Moderate score change observed")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Wanda score changes for safety-critical neurons"
    )

    parser.add_argument(
        "--original_scores",
        type=str,
        required=True,
        help="Path to original Wanda scores (.pt file)"
    )
    parser.add_argument(
        "--fine_tuned_scores",
        type=str,
        required=True,
        help="Path to fine-tuned Wanda scores (.pt file)"
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="Path to safety neuron masks (.pt file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Directory to save plots (default: ./plots)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading scores and masks...")
    original_scores = load_scores(args.original_scores)
    fine_tuned_scores = load_scores(args.fine_tuned_scores)
    masks = torch.load(args.masks)

    print(f"Original scores: {len(original_scores)} layers")
    print(f"Fine-tuned scores: {len(fine_tuned_scores)} layers")
    print(f"Masks: {len(masks)} layers")

    # Generate plots
    print("\nGenerating plots...")
    plot_score_distributions(original_scores, fine_tuned_scores, masks, args.output_dir)
    plot_score_comparison_stats(original_scores, fine_tuned_scores, masks, args.output_dir)

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
