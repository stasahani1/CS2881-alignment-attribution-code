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
import heapq
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

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
    snip_score_dir = os.path.join(score_dir, dataset_name, "wanda_score")

    if not os.path.exists(snip_score_dir):
        raise FileNotFoundError(f"Score directory not found: {snip_score_dir}")

    scores = {}
    score_files = sorted(Path(snip_score_dir).glob("W_metric_layer_*.pkl"))

    print(f"Loading SNIP scores from {snip_score_dir}")
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

        # Strip redundant "model.layers.{i}." prefix for compatibility with drift tracking
        # SNIP scores have full paths like "model.layers.0.self_attn.q_proj"
        # But we want keys like "layer_0_self_attn.q_proj"
        module_name = module_name.replace(f"model.layers.{layer_idx}.", "")

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
    Get top-k% neurons per layer (matches original paper methodology).

    Args:
        scores: Dictionary mapping layer_name to score tensor
        k: Fraction of neurons to select per layer (e.g., 0.01 for top 1%)

    Returns:
        List of (layer_name, row, col) tuples for top neurons
    """
    result = []
    total_neurons = 0

    print(f"Selecting top {k*100:.1f}% neurons per layer...")

    for layer_name, score in tqdm(scores.items(), desc="Processing layers"):
        # Ensure tensor is on CPU for memory efficiency
        if score.is_cuda:
            score = score.cpu()

        # Flatten and convert to numpy (convert to float32 first - bfloat16 not supported)
        flat_scores = score.flatten().float().numpy()

        # Calculate top-k for this layer
        layer_k = max(1, int(len(flat_scores) * k))
        layer_k = min(layer_k, len(flat_scores))

        if layer_k == 0:
            continue

        # Get top-k indices for this layer using argpartition (O(n))
        if layer_k < len(flat_scores):
            topk_indices = np.argpartition(flat_scores, -layer_k)[-layer_k:]
        else:
            topk_indices = np.arange(len(flat_scores))

        # Convert flat indices to row/col
        rows = topk_indices // score.shape[1]
        cols = topk_indices % score.shape[1]

        # Add to result
        for row, col in zip(rows, cols):
            result.append((layer_name, int(row), int(col)))

        total_neurons += len(topk_indices)

    print(f"Selected {total_neurons:,} neurons total (top {k*100:.1f}% per layer)")

    return result


def get_set_difference_neurons(
    safety_scores: Dict[str, torch.Tensor],
    utility_scores: Dict[str, torch.Tensor],
    p: float,
    q: float
) -> List[Tuple[str, int, int]]:
    """
    Get neurons in top-q% safety but NOT in top-p% utility (set difference).
    Uses layer-wise processing for memory efficiency and speed.

    Args:
        safety_scores: SNIP scores on safety dataset
        utility_scores: SNIP scores on utility dataset
        p: Fraction for utility threshold
        q: Fraction for safety threshold

    Returns:
        List of (layer_name, row, col) tuples for set difference neurons
    """
    result = []
    total_safety = 0
    total_utility = 0
    total_diff = 0

    print(f"Computing set difference (top-{q*100:.1f}% safety - top-{p*100:.1f}% utility)...")
    print("Processing layer-by-layer for efficiency...")

    for layer_name in tqdm(sorted(safety_scores.keys()), desc="Set difference"):
        if layer_name not in utility_scores:
            print(f"Warning: {layer_name} not in utility scores, skipping")
            continue

        safety_score = safety_scores[layer_name]
        utility_score = utility_scores[layer_name]

        # Ensure tensors are on CPU
        if safety_score.is_cuda:
            safety_score = safety_score.cpu()
        if utility_score.is_cuda:
            utility_score = utility_score.cpu()

        # Calculate counts for this layer
        total = safety_score.numel()
        top_p = int(total * p)
        top_q = int(total * q)

        # Flatten scores (convert to float32 first - bfloat16 not supported by numpy)
        flat_safety = safety_score.flatten().float().numpy()
        flat_utility = utility_score.flatten().float().numpy()

        # Get top indices (using argpartition for O(n) performance)
        if top_p > 0 and top_p < len(flat_utility):
            top_p_indices = np.argpartition(flat_utility, -top_p)[-top_p:]
        elif top_p >= len(flat_utility):
            top_p_indices = np.arange(len(flat_utility))
        else:
            top_p_indices = np.array([], dtype=np.int64)

        if top_q > 0 and top_q < len(flat_safety):
            top_q_indices = np.argpartition(flat_safety, -top_q)[-top_q:]
        elif top_q >= len(flat_safety):
            top_q_indices = np.arange(len(flat_safety))
        else:
            top_q_indices = np.array([], dtype=np.int64)

        # Set difference: elements in top_q but not in top_p
        # Note: indices are already unique (they're positions in the flattened array)
        mask = ~np.isin(top_q_indices, top_p_indices)
        filtered_indices = top_q_indices[mask]

        total_safety += len(top_q_indices)
        total_utility += len(top_p_indices)
        total_diff += len(filtered_indices)

        # Convert to row/col
        rows = filtered_indices // safety_score.shape[1]
        cols = filtered_indices % safety_score.shape[1]

        # Add to result
        for row, col in zip(rows, cols):
            result.append((layer_name, int(row), int(col)))

    print(f"Set difference: {total_diff} neurons")
    print(f"  (Top-{q*100:.1f}% safety: {total_safety}, "
          f"Top-{p*100:.1f}% utility: {total_utility})")

    return result


def get_random_neurons_per_layer(
    scores: Dict[str, torch.Tensor],
    k: float,
    seed: int = 0
) -> List[Tuple[str, int, int]]:
    """
    Get random sample of neurons per layer (matching SNIP/utility methodology).

    This samples k% of neurons from EACH layer independently, ensuring
    consistent methodology with safety and utility neuron identification.

    Args:
        scores: Dictionary mapping layer_name to score tensor (for dimensions)
        k: Fraction of neurons to sample per layer (e.g., 0.01 for 1%)
        seed: Random seed

    Returns:
        List of (layer_name, row, col) tuples for random neurons
    """
    np.random.seed(seed)
    result = []
    total_neurons = 0

    print(f"Selecting random {k*100:.1f}% neurons per layer...")

    for layer_name, score in tqdm(sorted(scores.items()), desc="Sampling random neurons"):
        # Ensure tensor is on CPU for memory efficiency
        if score.is_cuda:
            score = score.cpu()

        # Calculate how many neurons to sample from this layer
        layer_size = score.numel()
        layer_k = max(1, int(layer_size * k))
        layer_k = min(layer_k, layer_size)

        if layer_k == 0:
            continue

        # Sample random indices for this layer only
        random_indices = np.random.choice(layer_size, layer_k, replace=False)

        # Convert flat indices to row/col
        rows = random_indices // score.shape[1]
        cols = random_indices % score.shape[1]

        # Add to result
        for row, col in zip(rows, cols):
            result.append((layer_name, int(row), int(col)))

        total_neurons += len(random_indices)

    print(f"Selected {total_neurons:,} random neurons total ({k*100:.1f}% per layer)")

    return result


def save_neuron_scores(
    neurons: List[Tuple[str, int, int]],
    scores_dict: Dict[str, torch.Tensor],
    output_path: str
):
    """
    Save SNIP scores for identified neurons to a separate JSON file.
    
    Format: {
        "layer_0_self_attn.q_proj": [[row, col, score], ...],
        "layer_1_mlp.down_proj": [[row, col, score], ...],
        ...
    }
    
    Args:
        neurons: List of (layer_name, row, col) tuples
        scores_dict: Dictionary mapping layer_name to score tensor
        output_path: Path to save scores JSON file
    """
    # Group by layer first for batch processing
    layer_neurons = {}
    for layer_name, row, col in neurons:
        if layer_name not in layer_neurons:
            layer_neurons[layer_name] = []
        layer_neurons[layer_name].append((row, col))
    
    # Process each layer in batches (much faster than individual indexing)
    grouped = {}
    for layer_name, neuron_coords in tqdm(layer_neurons.items(), desc="Extracting scores"):
        if layer_name not in scores_dict:
            # If layer not found, save with score 0 as fallback
            grouped[layer_name] = [[row, col, 0.0] for row, col in neuron_coords]
            continue
        
        score_tensor = scores_dict[layer_name]
        if score_tensor.is_cuda:
            score_tensor = score_tensor.cpu()
        
        # Batch extract scores using advanced indexing (much faster)
        rows = [row for row, col in neuron_coords]
        cols = [col for row, col in neuron_coords]
        
        # Convert to tensors for batch indexing
        rows_tensor = torch.tensor(rows, dtype=torch.long)
        cols_tensor = torch.tensor(cols, dtype=torch.long)
        
        # Batch extract all scores at once - this is orders of magnitude faster
        scores_batch = score_tensor[rows_tensor, cols_tensor].float().numpy()
        
        # Store results
        grouped[layer_name] = [[row, col, float(score)] 
                               for (row, col), score in zip(neuron_coords, scores_batch)]
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(grouped, f, indent=2)
    
    print(f"Saved neuron scores to {output_path}")
    print(f"  Total layers: {len(grouped)}")
    print(f"  Total neurons with scores: {sum(len(v) for v in grouped.values())}")


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
        "--safety_dataset",
        type=str,
        default="align",
        help="Safety dataset name"
    )
    parser.add_argument(
        "--utility_dataset",
        type=str,
        default="alpaca_cleaned_no_safety",
        help="Utility dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="neuron_groups",
        help="Output directory for neuron group files"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["top_safety", "top_utility", "set_difference", "random"],
        default=["top_safety", "top_utility", "set_difference", "random"],
        help="Method(s) to run: top_safety, top_utility, set_difference, random (default: all)"
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
    
    # Convert method to set for easier checking
    methods = set(args.methods)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Identifying Neuron Groups from SNIP Scores")
    print("=" * 60)
    print(f"Score base directory: {args.score_base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Methods: {', '.join(sorted(methods))}")
    print(f"SNIP top-k: {args.snip_top_k}")
    print(f"Set diff p: {args.set_diff_p}, q: {args.set_diff_q}")
    print()

    # Determine which scores to load
    need_safety = "top_safety" in methods or "set_difference" in methods or "random" in methods
    need_utility = "top_utility" in methods or "set_difference" in methods

    # Load SNIP scores (only if needed)
    safety_scores = None
    utility_scores = None
    
    if need_safety:
        print("Loading safety SNIP scores...")
        safety_scores = load_snip_scores(args.score_base_dir, args.safety_dataset)
        print()

    if need_utility:
        print("Loading utility SNIP scores...")
        utility_scores = load_snip_scores(args.score_base_dir, args.utility_dataset)
        print()

    # Verify same layers in both (if both are loaded)
    if safety_scores is not None and utility_scores is not None:
        if set(safety_scores.keys()) != set(utility_scores.keys()):
            print("WARNING: Safety and utility scores have different layers!")
            print(f"  Safety layers: {len(safety_scores)}")
            print(f"  Utility layers: {len(utility_scores)}")

    # Track results for summary
    results = {}

    # 1. Safety-critical (SNIP top-k)
    if "top_safety" in methods:
        print("=" * 60)
        print("1. Identifying safety-critical neurons (SNIP top-k)...")
        print("=" * 60)
        safety_topk = get_topk_neurons(safety_scores, args.snip_top_k)
        save_neuron_scores(
            safety_topk,
            safety_scores,
            os.path.join(args.output_dir, "neuron_groups_top_safety.json")
        )
        results["top_safety"] = len(safety_topk)
        print()

    # 2. Safety-critical (Set difference)
    if "set_difference" in methods:
        print("=" * 60)
        print("2. Identifying safety-critical neurons (Set difference)...")
        print("=" * 60)
        set_diff_neurons = get_set_difference_neurons(
            safety_scores, utility_scores, args.set_diff_p, args.set_diff_q
        )
        save_neuron_scores(
            set_diff_neurons,
            safety_scores,
            os.path.join(args.output_dir, "neuron_groups_set_diff.json")
        )
        results["set_difference"] = len(set_diff_neurons)
        print()

    # 3. Utility-critical
    if "top_utility" in methods:
        print("=" * 60)
        print("3. Identifying utility-critical neurons...")
        print("=" * 60)
        utility_topk = get_topk_neurons(utility_scores, args.snip_top_k)
        save_neuron_scores(
            utility_topk,
            utility_scores,
            os.path.join(args.output_dir, "neuron_groups_top_utility.json")
        )
        results["top_utility"] = len(utility_topk)
        print()

    # 4. Random (same percentage per layer as safety SNIP top-k)
    if "random" in methods:
        print("=" * 60)
        print("4. Identifying random neurons (baseline)...")
        print("=" * 60)
        random_neurons = get_random_neurons_per_layer(
            safety_scores, args.snip_top_k, args.seed
        )
        save_neuron_scores(
            random_neurons,
            safety_scores,
            os.path.join(args.output_dir, "neuron_groups_random.json")
        )
        results["random"] = len(random_neurons)
        print()

    print("=" * 60)
    print("Neuron Group Identification Complete!")
    print("=" * 60)
    print(f"Output files in {args.output_dir}:")
    if "top_safety" in results:
        print(f"  - neuron_groups_snip_top.json    ({results['top_safety']:,} neurons)")
    if "set_difference" in results:
        print(f"  - neuron_groups_set_difference.json    ({results['set_difference']:,} neurons)")
    if "top_utility" in results:
        print(f"  - neuron_groups_utility.json     ({results['top_utility']:,} neurons)")
    if "random" in results:
        print(f"  - neuron_groups_random.json      ({results['random']:,} neurons)")


if __name__ == "__main__":
    main()