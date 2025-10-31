"""
Identify neuron groups from SNIP scores.

This script loads pre-computed SNIP scores and identifies:
1. Safety-critical neurons (SNIP top-k method)
2. Safety-critical neurons (Set difference method)
3. Utility-critical neurons
4. Random neurons (baseline)

Only neuron IDs are saved, not the scores themselves.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def load_snip_scores(score_dir: str, dataset_name: str) -> Dict[str, torch.Tensor]:
    """
    Load SNIP scores from directory.

    Args:
        score_dir: Base directory containing SNIP scores
        dataset_name: Name of dataset (align or alpaca_cleaned_no_safety)

    Returns:
        Dictionary mapping layer_name to score tensor
    """
    wanda_score_dir = os.path.join(score_dir, dataset_name, "wanda_score")

    if not os.path.exists(wanda_score_dir):
        raise FileNotFoundError(f"Score directory not found: {wanda_score_dir}")

    scores = {}
    score_files = sorted(Path(wanda_score_dir).glob("W_metric_layer_*.pkl"))

    print(f"Loading SNIP scores from {wanda_score_dir}")
    print(f"Found {len(score_files)} score files")

    for score_file in score_files:
        # Parse filename: W_metric_layer_{i}_name_{name}_weight.pkl
        filename = score_file.stem
        parts = filename.split("_")

        # Extract layer index and module name
        layer_idx = None
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                break

        # Extract module name (everything between "name_" and "_weight")
        name_start = filename.find("name_") + 5
        name_end = filename.rfind("_weight")
        if name_end == -1:
            name_end = len(filename)
        module_name = filename[name_start:name_end]

        # Create unique key
        key = f"layer_{layer_idx}_{module_name}"

        # Load score tensor
        with open(score_file, "rb") as f:
            score = pickle.load(f)
            scores[key] = score

        print(f"  Loaded {key}: shape {score.shape}")

    return scores


def get_topk_neurons(
    scores: Dict[str, torch.Tensor],
    k: float
) -> List[Tuple[str, int, int]]:
    """
    Get top-k% neurons by score.

    Args:
        scores: Dictionary mapping layer_name to score tensor
        k: Fraction of neurons to select (e.g., 0.01 for top 1%)

    Returns:
        List of (layer_name, row, col) tuples for top neurons
    """
    # Flatten all scores and track indices
    all_scores = []
    all_indices = []

    for layer_name, score in scores.items():
        flat_score = score.flatten()
        for idx in range(len(flat_score)):
            row = idx // score.shape[1]
            col = idx % score.shape[1]
            all_scores.append(flat_score[idx].item())
            all_indices.append((layer_name, row, col))

    # Sort and get top-k%
    num_topk = int(len(all_scores) * k)
    topk_idx = np.argsort(all_scores)[-num_topk:]
    topk_neurons = [all_indices[i] for i in topk_idx]

    print(f"Selected {len(topk_neurons)} neurons (top {k*100:.1f}%)")

    return topk_neurons


def get_set_difference_neurons(
    safety_scores: Dict[str, torch.Tensor],
    utility_scores: Dict[str, torch.Tensor],
    p: float,
    q: float
) -> List[Tuple[str, int, int]]:
    """
    Get neurons in top-q% safety but NOT in top-p% utility (set difference).

    Args:
        safety_scores: SNIP scores on safety dataset
        utility_scores: SNIP scores on utility dataset
        p: Fraction for utility threshold
        q: Fraction for safety threshold

    Returns:
        List of (layer_name, row, col) tuples for set difference neurons
    """
    # Get top-p% utility neurons
    print(f"Computing top-{p*100:.1f}% utility neurons...")
    utility_topk = set(get_topk_neurons(utility_scores, p))

    # Get top-q% safety neurons
    print(f"Computing top-{q*100:.1f}% safety neurons...")
    safety_topk = set(get_topk_neurons(safety_scores, q))

    # Set difference: safety - utility
    set_diff = safety_topk - utility_topk

    print(f"Set difference: {len(set_diff)} neurons")
    print(f"  (Top-{q*100:.1f}% safety: {len(safety_topk)}, "
          f"Top-{p*100:.1f}% utility: {len(utility_topk)})")

    return list(set_diff)


def get_random_neurons(
    scores: Dict[str, torch.Tensor],
    num_neurons: int,
    seed: int = 0
) -> List[Tuple[str, int, int]]:
    """
    Get random sample of neurons.

    Args:
        scores: Dictionary mapping layer_name to score tensor (for dimensions)
        num_neurons: Number of neurons to sample
        seed: Random seed

    Returns:
        List of (layer_name, row, col) tuples for random neurons
    """
    np.random.seed(seed)

    # Get all possible neuron indices
    all_indices = []
    for layer_name, score in scores.items():
        for row in range(score.shape[0]):
            for col in range(score.shape[1]):
                all_indices.append((layer_name, row, col))

    # Random sample
    sampled_idx = np.random.choice(len(all_indices), num_neurons, replace=False)
    random_neurons = [all_indices[i] for i in sampled_idx]

    print(f"Sampled {len(random_neurons)} random neurons")

    return random_neurons


def save_neuron_groups(
    neurons: List[Tuple[str, int, int]],
    output_path: str
):
    """
    Save neuron groups to JSON file.

    Format: {
        "layer_0_self_attn.q_proj": [[row, col], ...],
        "layer_1_mlp.down_proj": [[row, col], ...],
        ...
    }
    """
    # Group by layer
    grouped = {}
    for layer_name, row, col in neurons:
        if layer_name not in grouped:
            grouped[layer_name] = []
        grouped[layer_name].append([row, col])

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(grouped, f, indent=2)

    print(f"Saved neuron groups to {output_path}")
    print(f"  Total layers: {len(grouped)}")
    print(f"  Total neurons: {sum(len(v) for v in grouped.values())}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify neuron groups from SNIP scores"
    )
    parser.add_argument(
        "--score_base_dir",
        type=str,
        default="/dev/shm/snip_scores",
        help="Base directory containing SNIP scores"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="neuron_groups",
        help="Output directory for neuron group files"
    )
    parser.add_argument(
        "--snip_top_k",
        type=float,
        default=0.01,
        help="Top-k fraction for SNIP method (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--set_diff_p",
        type=float,
        default=0.1,
        help="Top-p%% utility neurons for set difference (default: 0.1)"
    )
    parser.add_argument(
        "--set_diff_q",
        type=float,
        default=0.1,
        help="Top-q%% safety neurons for set difference (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for random neuron selection"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Identifying Neuron Groups from SNIP Scores")
    print("=" * 60)
    print(f"Score base directory: {args.score_base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"SNIP top-k: {args.snip_top_k}")
    print(f"Set diff p: {args.set_diff_p}, q: {args.set_diff_q}")
    print()

    # Load SNIP scores
    print("Loading safety SNIP scores...")
    safety_scores = load_snip_scores(args.score_base_dir, "align")
    print()

    print("Loading utility SNIP scores...")
    utility_scores = load_snip_scores(args.score_base_dir, "alpaca_cleaned_no_safety")
    print()

    # Verify same layers in both
    if set(safety_scores.keys()) != set(utility_scores.keys()):
        print("WARNING: Safety and utility scores have different layers!")
        print(f"  Safety layers: {len(safety_scores)}")
        print(f"  Utility layers: {len(utility_scores)}")

    # 1. Safety-critical (SNIP top-k)
    print("=" * 60)
    print("1. Identifying safety-critical neurons (SNIP top-k)...")
    print("=" * 60)
    safety_topk = get_topk_neurons(safety_scores, args.snip_top_k)
    save_neuron_groups(
        safety_topk,
        os.path.join(args.output_dir, "neuron_groups_snip_top.json")
    )
    print()

    # 2. Safety-critical (Set difference)
    print("=" * 60)
    print("2. Identifying safety-critical neurons (Set difference)...")
    print("=" * 60)
    set_diff_neurons = get_set_difference_neurons(
        safety_scores, utility_scores, args.set_diff_p, args.set_diff_q
    )
    save_neuron_groups(
        set_diff_neurons,
        os.path.join(args.output_dir, "neuron_groups_set_diff.json")
    )
    print()

    # 3. Utility-critical
    print("=" * 60)
    print("3. Identifying utility-critical neurons...")
    print("=" * 60)
    utility_topk = get_topk_neurons(utility_scores, args.snip_top_k)
    save_neuron_groups(
        utility_topk,
        os.path.join(args.output_dir, "neuron_groups_utility.json")
    )
    print()

    # 4. Random (same size as safety SNIP top-k)
    print("=" * 60)
    print("4. Identifying random neurons (baseline)...")
    print("=" * 60)
    random_neurons = get_random_neurons(
        safety_scores, len(safety_topk), args.seed
    )
    save_neuron_groups(
        random_neurons,
        os.path.join(args.output_dir, "neuron_groups_random.json")
    )
    print()

    print("=" * 60)
    print("Neuron Group Identification Complete!")
    print("=" * 60)
    print(f"Output files in {args.output_dir}:")
    print(f"  - neuron_groups_snip_top.json    ({len(safety_topk)} neurons)")
    print(f"  - neuron_groups_set_diff.json    ({len(set_diff_neurons)} neurons)")
    print(f"  - neuron_groups_utility.json     ({len(utility_topk)} neurons)")
    print(f"  - neuron_groups_random.json      ({len(random_neurons)} neurons)")


if __name__ == "__main__":
    main()
