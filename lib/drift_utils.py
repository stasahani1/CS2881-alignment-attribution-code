"""
Utilities for tracking and computing neuron weight drift during fine-tuning.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


def load_neuron_groups(neuron_groups_dir: str) -> Dict[str, Dict]:
    """
    Load all neuron group files.

    Args:
        neuron_groups_dir: Directory containing neuron group JSON files

    Returns:
        Dictionary mapping group name to neuron coordinates
    """
    groups = {}
    group_files = {
        "safety_snip_top": "neuron_groups_snip_top.json",
        "safety_set_diff": "neuron_groups_set_diff.json",
        "utility": "neuron_groups_utility.json",
        # "random": "neuron_groups_random.json",
    }

    for group_name, filename in group_files.items():
        filepath = os.path.join(neuron_groups_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                groups[group_name] = json.load(f)
            print(f"Loaded {group_name}: {sum(len(v) for v in groups[group_name].values())} neurons")
        else:
            print(f"Warning: {filepath} not found, skipping {group_name}")

    # Validate that at least one group was loaded
    if not groups:
        raise ValueError(
            f"No neuron groups loaded from {neuron_groups_dir}! "
            f"Make sure you've run phase 1 (identify_neuron_groups.py) first."
        )

    # Validate that groups are not empty
    total_neurons = sum(sum(len(v) for v in g.values()) for g in groups.values())
    if total_neurons == 0:
        raise ValueError(
            f"All neuron groups are empty! Check that phase 1 completed successfully."
        )

    return groups


def extract_neuron_weights(
    model: torch.nn.Module,
    neuron_groups: Dict[str, List[Tuple[int, int]]],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Extract weight vectors for specific neurons from model.

    Optimized version using batch extraction per layer for ~10-20x speedup.

    Args:
        model: PyTorch model
        neuron_groups: Dict mapping layer_name to list of [row, col] coordinates
        device: Device to store tensors

    Returns:
        Dictionary mapping (layer_name, row, col) to weight value
    """
    weights = {}
    total_neurons = sum(len(coords) for coords in neuron_groups.values())

    print(f"Extracting weights for {total_neurons:,} neurons across {len(neuron_groups)} layers...")

    with tqdm(total=total_neurons, desc="Extracting weights") as pbar:
        for layer_name, neuron_coords in neuron_groups.items():
            # Skip empty coordinate lists
            if not neuron_coords:
                continue

            # Parse layer name: "layer_0_self_attn.q_proj" -> layer 0, module self_attn.q_proj
            parts = layer_name.split("_", 2)
            if len(parts) >= 3 and parts[0] == "layer":
                layer_idx = int(parts[1])
                module_path = parts[2]

                # Navigate to the layer
                try:
                    layer = model.model.layers[layer_idx]

                    # Navigate to the specific module (e.g., self_attn.q_proj)
                    module = layer
                    for attr in module_path.split("."):
                        module = getattr(module, attr)

                    # Extract weight matrix
                    weight_matrix = module.weight.data

                    # Batch extract all neurons from this layer using advanced indexing
                    # Convert coordinates to numpy arrays for efficiency
                    coords_array = np.array(neuron_coords)
                    rows = coords_array[:, 0]
                    cols = coords_array[:, 1]

                    # Bounds checking for all coordinates
                    valid_mask = (rows < weight_matrix.shape[0]) & (cols < weight_matrix.shape[1])
                    invalid_count = (~valid_mask).sum()

                    if invalid_count > 0:
                        print(f"Warning: {invalid_count} coordinates out of bounds for {layer_name} shape {weight_matrix.shape}, skipping")
                        rows = rows[valid_mask]
                        cols = cols[valid_mask]

                    # Extract all weights at once (single GPU operation)
                    if len(rows) > 0:
                        layer_weights = weight_matrix[rows, cols].float().cpu()

                        # Store in dictionary with keys
                        for idx, (row, col) in enumerate(zip(rows, cols)):
                            key = f"{layer_name}_{row}_{col}"
                            weights[key] = layer_weights[idx].clone()

                    pbar.update(len(neuron_coords))

                except (AttributeError, IndexError) as e:
                    print(f"Warning: Could not access {layer_name}: {e}")
                    pbar.update(len(neuron_coords))
                    continue

    # Report memory usage
    total_weights = len(weights)
    memory_mb = sum(w.element_size() * w.nelement() for w in weights.values()) / (1024 * 1024)
    print(f"Extracted {total_weights:,} weights, using {memory_mb:.2f} MB")

    return weights


def compute_absolute_change(
    initial_weights: Dict[str, torch.Tensor],
    current_weights: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute absolute change |current - initial| for each weight.

    This is more appropriate than cosine similarity for scalar weights,
    as it captures magnitude changes directly.

    Args:
        initial_weights: Initial weight values
        current_weights: Current weight values

    Returns:
        Dictionary mapping neuron key to absolute change
    """
    changes = {}

    for key in initial_weights.keys():
        if key in current_weights:
            initial = initial_weights[key]
            current = current_weights[key]

            # Compute absolute difference
            abs_change = torch.abs(current - initial).item()
            changes[key] = abs_change

    return changes


def compute_l2_distance(
    initial_weights: Dict[str, torch.Tensor],
    current_weights: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute L2 distance between initial and current weights.

    Args:
        initial_weights: Initial weight values
        current_weights: Current weight values

    Returns:
        Dictionary mapping neuron key to L2 distance
    """
    distances = {}

    for key in initial_weights.keys():
        if key in current_weights:
            initial = initial_weights[key]
            current = current_weights[key]
            l2_dist = torch.norm(current - initial, p=2).item()
            distances[key] = l2_dist

    return distances


def compute_relative_change(
    initial_weights: Dict[str, torch.Tensor],
    current_weights: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute relative change: |current - initial| / |initial|

    For scalar weights, this is simply the percentage change.
    For weight vectors, this uses L2 norms.

    Args:
        initial_weights: Initial weight values
        current_weights: Current weight values

    Returns:
        Dictionary mapping neuron key to relative change
    """
    relative_changes = {}

    for key in initial_weights.keys():
        if key in current_weights:
            initial = initial_weights[key]
            current = current_weights[key]

            # Compute magnitudes
            initial_mag = torch.abs(initial).item() if initial.numel() == 1 else torch.norm(initial, p=2).item()

            if initial_mag > 1e-8:  # Avoid division by zero
                # Compute change magnitude
                change_mag = torch.abs(current - initial).item() if initial.numel() == 1 else torch.norm(current - initial, p=2).item()
                relative_changes[key] = change_mag / initial_mag
            else:
                # Undefined for near-zero initial weights
                relative_changes[key] = 0.0

    return relative_changes


def aggregate_by_group(
    metrics: Dict[str, float],
    neuron_groups: Dict[str, Dict]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics by neuron group.

    Args:
        metrics: Dictionary mapping neuron key to metric value
        neuron_groups: Dictionary mapping group name to neuron coordinates

    Returns:
        Dictionary mapping group name to statistics (mean, std, min, max, median)
    """
    aggregated = {}

    for group_name, group_neurons in neuron_groups.items():
        # Collect metrics for this group
        group_values = []

        for layer_name, neuron_coords in group_neurons.items():
            for row, col in neuron_coords:
                key = f"{layer_name}_{row}_{col}"
                if key in metrics:
                    group_values.append(metrics[key])

        # Compute statistics
        if group_values:
            aggregated[group_name] = {
                "mean": float(np.mean(group_values)),
                "std": float(np.std(group_values)),
                "min": float(np.min(group_values)),
                "max": float(np.max(group_values)),
                "median": float(np.median(group_values)),
                "q25": float(np.percentile(group_values, 25)),
                "q75": float(np.percentile(group_values, 75)),
                "count": len(group_values),
            }
        else:
            aggregated[group_name] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "q25": 0.0,
                "q75": 0.0,
                "count": 0,
            }

    return aggregated


def compute_drift_metrics(
    model: torch.nn.Module,
    initial_weights: Dict[str, torch.Tensor],
    neuron_groups: Dict[str, Dict],
    device: str = "cuda"
) -> Dict[str, Dict]:
    """
    Compute all drift metrics for current model state.

    Args:
        model: Current model
        initial_weights: Initial weights before fine-tuning
        neuron_groups: Neuron group definitions
        device: Device

    Returns:
        Dictionary containing aggregated metrics by group
    """
    # Extract current weights for all tracked neurons
    # Merge all neuron groups into a single dict, combining coords for overlapping layers
    all_neurons = {}
    for group_name, group_data in neuron_groups.items():
        for layer_name, coords in group_data.items():
            if layer_name not in all_neurons:
                all_neurons[layer_name] = []
            all_neurons[layer_name].extend(coords)

    current_weights = extract_neuron_weights(model, all_neurons, device)

    # Compute individual metrics
    # Use absolute and relative change instead of cosine similarity
    # (more appropriate for scalar weights and LoRA modifications)
    absolute_changes = compute_absolute_change(initial_weights, current_weights)
    relative_changes = compute_relative_change(initial_weights, current_weights)

    # Aggregate by group
    results = {
        "absolute_change": aggregate_by_group(absolute_changes, neuron_groups),
        "relative_change": aggregate_by_group(relative_changes, neuron_groups),
    }

    return results


def save_drift_log(
    drift_metrics: Dict,
    step: int,
    output_dir: str
):
    """
    Save drift metrics for a checkpoint.

    Args:
        drift_metrics: Computed drift metrics
        step: Training step number
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"drift_step_{step:06d}.json")

    # Add metadata
    log_data = {
        "step": step,
        "metrics": drift_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"Saved drift log to {output_path}")
