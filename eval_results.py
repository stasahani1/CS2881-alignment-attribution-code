"""
Evaluation script to compare neuron groups between original and finetuned models.
Calculates the percentage of neurons that are in both set_diffs and top_safety, layerwise and overall.
"""

import json
import os
import re
from typing import Dict, Set, Tuple, List


def load_neuron_group(filepath: str) -> Dict[str, list]:
    """
    Load a neuron group JSON file.
    
    Format: {
        "layer_name": [[row, col], ...],
        ...
    }
    
    Args:
        filepath: Path to the neuron group JSON file
        
    Returns:
        Dictionary mapping layer names to lists of [row, col] neuron coordinates
    """
    with open(filepath, "r") as f:
        return json.load(f)


def normalize_layer_name(layer_name: str) -> str:
    """
    Normalize layer names to handle different naming conventions.
    
    Original format: layer_0_mlp.down_proj
    Finetuned format: layer_0_base_model.model.mlp.down_proj
    
    Normalized format: layer_0_mlp.down_proj (standardize to original format)
    
    Args:
        layer_name: Original layer name
        
    Returns:
        Normalized layer name
    """
    # Pattern to match layer_<number>_<rest>
    match = re.match(r'(layer_\d+)_(.+)', layer_name)
    if not match:
        return layer_name  # Return as-is if doesn't match expected pattern
    
    layer_prefix, component = match.groups()
    
    # Remove "base_model.model." prefix if present (from finetuned format)
    if component.startswith('base_model.model.'):
        component = component.replace('base_model.model.', '', 1)
    
    return f"{layer_prefix}_{component}"


def neuron_list_to_set(neurons: List[List[int]]) -> Set[Tuple[int, int]]:
    """
    Convert a list of [row, col] neurons to a set of (row, col) tuples.
    
    Args:
        neurons: List of [row, col] neuron coordinates
        
    Returns:
        Set of (row, col) tuples
    """
    return set((row, col) for row, col in neurons)


def calculate_overlap_percentage(
    original_neurons: Set[Tuple[int, int]],
    finetuned_neurons: Set[Tuple[int, int]]
) -> float:
    """
    Calculate the percentage of original neurons that are also in finetuned set.
    
    Args:
        original_neurons: Set of neurons from original model
        finetuned_neurons: Set of neurons from finetuned model
        
    Returns:
        Percentage (0-100) of original neurons that are in finetuned set
    """
    if len(original_neurons) == 0:
        return 0.0
    
    intersection = original_neurons & finetuned_neurons
    percentage = (len(intersection) / len(original_neurons)) * 100.0
    return percentage


def analyze_neuron_group(
    group_name: str,
    original_path: str,
    finetuned_path: str,
    output_dir: str
):
    """
    Analyze overlap between original and finetuned neuron groups.
    
    Args:
        group_name: Name of the neuron group (e.g., "set_diff", "top_safety")
        original_path: Path to original neuron group file
        finetuned_path: Path to finetuned neuron group file
        output_dir: Directory to save output files
    """
    # Validate files exist
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original {group_name} file not found: {original_path}")
    if not os.path.exists(finetuned_path):
        raise FileNotFoundError(f"Finetuned {group_name} file not found: {finetuned_path}")
    
    print("=" * 80)
    print(f"Evaluating {group_name.upper().replace('_', ' ')} Neuron Group Overlap (Layerwise)")
    print("=" * 80)
    print()
    
    # Load neuron groups
    print(f"Loading original {group_name} neurons from: {original_path}")
    original_neuron_group_raw = load_neuron_group(original_path)
    print(f"Loading finetuned {group_name} neurons from: {finetuned_path}")
    finetuned_neuron_group_raw = load_neuron_group(finetuned_path)
    print()
    
    # Normalize layer names for both groups
    original_neuron_group = {}
    for layer_name, neurons in original_neuron_group_raw.items():
        normalized = normalize_layer_name(layer_name)
        if normalized in original_neuron_group:
            # Merge if duplicate normalized names (shouldn't happen, but be safe)
            original_neuron_group[normalized].extend(neurons)
        else:
            original_neuron_group[normalized] = neurons
    
    finetuned_neuron_group = {}
    for layer_name, neurons in finetuned_neuron_group_raw.items():
        normalized = normalize_layer_name(layer_name)
        if normalized in finetuned_neuron_group:
            # Merge if duplicate normalized names (shouldn't happen, but be safe)
            finetuned_neuron_group[normalized].extend(neurons)
        else:
            finetuned_neuron_group[normalized] = neurons
    
    # Get all layers (union of both groups)
    all_layers = sorted(set(original_neuron_group.keys()) | set(finetuned_neuron_group.keys()))
    
    # Calculate layerwise overlaps
    layerwise_results = []
    total_original = 0
    total_finetuned = 0
    total_overlap = 0
    
    print("=" * 80)
    print(f"Layerwise Overlap Results - {group_name.upper().replace('_', ' ')}")
    print("=" * 80)
    print(f"{'Layer':<50} {'Original':<12} {'Finetuned':<12} {'Overlap':<12} {'Percentage':<12}")
    print("-" * 80)
    
    for layer in all_layers:
        original_neurons = neuron_list_to_set(original_neuron_group.get(layer, []))
        finetuned_neurons = neuron_list_to_set(finetuned_neuron_group.get(layer, []))
        
        overlap = original_neurons & finetuned_neurons
        overlap_count = len(overlap)
        overlap_pct = calculate_overlap_percentage(original_neurons, finetuned_neurons)
        
        original_count = len(original_neurons)
        finetuned_count = len(finetuned_neurons)
        
        # Store for overall calculation
        total_original += original_count
        total_finetuned += finetuned_count
        total_overlap += overlap_count
        
        layerwise_results.append({
            'layer': layer,
            'original': original_count,
            'finetuned': finetuned_count,
            'overlap': overlap_count,
            'percentage': round(overlap_pct, 2)
        })
        
        # Print layer result
        print(f"{layer:<50} {original_count:<12,} {finetuned_count:<12,} {overlap_count:<12,} {overlap_pct:>10.2f}%")
    
    # Calculate overall percentage
    overall_percentage = (total_overlap / total_original * 100.0) if total_original > 0 else 0.0
    
    print("-" * 80)
    print(f"{'TOTAL':<50} {total_original:<12,} {total_finetuned:<12,} {total_overlap:<12,} {overall_percentage:>10.2f}%")
    print("=" * 80)
    print()
    
    # Calculate additional statistics
    avg_layerwise_percentage = 0.0
    weighted_avg = 0.0
    
    if layerwise_results:
        layerwise_percentages = [r['percentage'] for r in layerwise_results if r['original'] > 0]
        if layerwise_percentages:
            avg_layerwise_percentage = sum(layerwise_percentages) / len(layerwise_percentages)
            weighted_avg = sum(
                r['percentage'] * r['original'] 
                for r in layerwise_results if r['original'] > 0
            ) / total_original if total_original > 0 else 0.0
    
    # Summary
    print("=" * 80)
    print(f"Summary - {group_name.upper().replace('_', ' ')}")
    print("=" * 80)
    print(f"Total layers analyzed:        {len(all_layers)}")
    print(f"Total original neurons:       {total_original:,}")
    print(f"Total finetuned neurons:      {total_finetuned:,}")
    print(f"Total neurons in both:        {total_overlap:,}")
    print(f"Overall overlap percentage:   {overall_percentage:.2f}%")
    print("=" * 80)
    
    # Additional statistics
    if layerwise_percentages:
        print()
        print("Additional Statistics")
        print("-" * 80)
        print(f"Average layerwise percentage (unweighted): {avg_layerwise_percentage:.2f}%")
        print(f"Average layerwise percentage (weighted):   {weighted_avg:.2f}%")
        print("=" * 80)
    
    # Save layerwise results to file
    layerwise_output_path = os.path.join(output_dir, f"layerwise_{group_name}.json")
    with open(layerwise_output_path, "w") as f:
        json.dump(layerwise_results, f, indent=2)
    print(f"\nSaved layerwise results to: {layerwise_output_path}")
    
    # Prepare aggregated results
    aggregated_results = {
        'group_name': group_name,
        'total_layers': len(all_layers),
        'total_original_neurons': total_original,
        'total_finetuned_neurons': total_finetuned,
        'total_overlap_neurons': total_overlap,
        'overall_overlap_percentage': round(overall_percentage, 2),
        'average_layerwise_percentage_unweighted': round(avg_layerwise_percentage, 2),
        'average_layerwise_percentage_weighted': round(weighted_avg, 2)
    }
    
    print()
    print()
    
    return aggregated_results


def main():
    # Create output directory
    output_dir = "outputs/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Neuron Group Overlap Evaluation")
    print("=" * 80)
    print()
    
    all_aggregated_results = {}
    
    # Analyze set_diff neurons
    original_set_diff_path = "outputs/neuron_groups/neuron_groups_set_diff.json"
    finetuned_set_diff_path = "outputs/neuron_groups_finetuned/neuron_groups_set_diff.json"
    aggregated_set_diff = analyze_neuron_group("set_diff", original_set_diff_path, finetuned_set_diff_path, output_dir)
    all_aggregated_results['set_diff'] = aggregated_set_diff
    
    # Analyze top_safety neurons
    original_top_safety_path = "outputs/neuron_groups/neuron_groups_top_safety.json"
    finetuned_top_safety_path = "outputs/neuron_groups_finetuned/neuron_groups_top_safety.json"
    aggregated_top_safety = analyze_neuron_group("top_safety", original_top_safety_path, finetuned_top_safety_path, output_dir)
    all_aggregated_results['top_safety'] = aggregated_top_safety
    
    # Save aggregated results to file
    aggregated_output_path = os.path.join(output_dir, "aggregated_results.json")
    with open(aggregated_output_path, "w") as f:
        json.dump(all_aggregated_results, f, indent=2)
    print(f"Saved aggregated results to: {aggregated_output_path}")
    
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()