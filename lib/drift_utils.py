"""
Utilities for tracking and computing neuron weight drift during fine-tuning.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
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
        "random": "neuron_groups_random.json",
    }

    for group_name, filename in group_files.items():
        filepath = os.path.join(neuron_groups_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                groups[group_name] = json.load(f)
            print(f"Loaded {group_name}: {sum(len(v) for v in groups[group_name].values())} neurons")
        else:
            print(f"Warning: {filepath} not found, skipping {group_name}")

    return groups


def extract_neuron_weights(
    model: torch.nn.Module,
    neuron_groups: Dict[str, List[Tuple[int, int]]],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Extract weight vectors for specific neurons from model.

    Args:
        model: PyTorch model
        neuron_groups: Dict mapping layer_name to list of [row, col] coordinates
        device: Device to store tensors

    Returns:
        Dictionary mapping (layer_name, row, col) to weight value
    """
    weights = {}

    for layer_name, neuron_coords in neuron_groups.items():
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

                # Extract specific neuron weights
                for row, col in neuron_coords:
                    key = f"{layer_name}_{row}_{col}"
                    # Store the full row (incoming weights to this neuron)
                    weights[key] = weight_matrix[row, col].clone().cpu()

            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not access {layer_name}: {e}")
                continue

    return weights


def compute_cosine_similarity(
    initial_weights: Dict[str, torch.Tensor],
    current_weights: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute cosine similarity between initial and current weights.

    Args:
        initial_weights: Initial weight values
        current_weights: Current weight values

    Returns:
        Dictionary mapping neuron key to cosine similarity
    """
    similarities = {}

    for key in initial_weights.keys():
        if key in current_weights:
            initial = initial_weights[key]
            current = current_weights[key]

            # Cosine similarity for scalar values is just sign agreement
            # For actual weight rows, use proper cosine similarity
            if initial.numel() == 1:
                # Single weight: just check if signs match
                similarities[key] = 1.0 if torch.sign(initial) == torch.sign(current) else -1.0
            else:
                # Weight vector: compute cosine similarity
                cos_sim = F.cosine_similarity(
                    initial.unsqueeze(0), current.unsqueeze(0)
                ).item()
                similarities[key] = cos_sim

    return similarities


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
    Compute relative change (L2 distance normalized by initial norm).

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

            initial_norm = torch.norm(initial, p=2).item()
            if initial_norm > 1e-8:  # Avoid division by zero
                l2_dist = torch.norm(current - initial, p=2).item()
                relative_changes[key] = l2_dist / initial_norm
            else:
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
    all_neurons = {}
    for group_name, group_data in neuron_groups.items():
        all_neurons.update(group_data)

    current_weights = extract_neuron_weights(model, all_neurons, device)

    # Compute individual metrics
    cosine_sims = compute_cosine_similarity(initial_weights, current_weights)
    l2_dists = compute_l2_distance(initial_weights, current_weights)
    relative_changes = compute_relative_change(initial_weights, current_weights)

    # Aggregate by group
    results = {
        "cosine_similarity": aggregate_by_group(cosine_sims, neuron_groups),
        "l2_distance": aggregate_by_group(l2_dists, neuron_groups),
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
