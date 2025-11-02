"""
Compute neuron weight drift for specified checkpoints.

This script:
1. Loads initial weights (pre-training baseline)
2. For each checkpoint:
   - Loads base model
   - Loads LoRA adapter from checkpoint
   - Merges LoRA into base model
   - Computes drift against initial weights
   - Saves drift log
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from lib.drift_utils import (
    load_neuron_groups,
    compute_drift_metrics,
    save_drift_log,
)


def load_checkpoint(
    base_model_path: str,
    checkpoint_path: str,
    device: str = "auto"
):
    """
    Load model from checkpoint with LoRA merge.

    Args:
        base_model_path: Path to base model (same as used in training)
        checkpoint_path: Path to checkpoint directory (e.g., checkpoint-500)
        device: Device map

    Returns:
        Merged model (base + LoRA)
    """
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )

    # Merge LoRA into base model
    model = model.merge_and_unload()

    return model


def extract_step_from_checkpoint(checkpoint_path: str) -> int:
    """Extract step number from checkpoint path."""
    # checkpoint_path format: "path/to/checkpoint-500" or "checkpoint-500"
    basename = os.path.basename(checkpoint_path.rstrip("/"))
    if basename.startswith("checkpoint-"):
        return int(basename.split("-")[1])

    # Fallback: try to read from trainer_state.json
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            state = json.load(f)
            return state.get("global_step", 0)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compute drift for specified checkpoints"
    )

    # Model arguments
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Path to base model (same as used in training)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoints (will process all checkpoint-* subdirs)"
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        default=None,
        help="Specific checkpoint paths to process (overrides --checkpoint_dir)"
    )

    # Drift tracking arguments
    parser.add_argument(
        "--neuron_groups_dir",
        type=str,
        default="/workspace/CS2881-alignment-attribution-code/neuron_groups",
        help="Directory containing neuron group JSON files"
    )
    parser.add_argument(
        "--initial_weights_path",
        type=str,
        default="/dev/shm/initial_weights/initial_weights.pt",
        help="Path to initial weights file (from finetune_with_tracking.py)"
    )
    parser.add_argument(
        "--drift_log_dir",
        type=str,
        default="/dev/shm/drift_logs",
        help="Directory to save drift logs"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Computing Neuron Weight Drift for Checkpoints")
    print("=" * 70)

    # Determine which checkpoints to process
    if args.checkpoint_paths:
        checkpoint_paths = args.checkpoint_paths
    elif args.checkpoint_dir:
        # Find all checkpoint-* directories
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_paths = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0
        )
        checkpoint_paths = [str(p) for p in checkpoint_paths]
    else:
        print("ERROR: Must specify either --checkpoint_dir or --checkpoint_paths")
        return

    if not checkpoint_paths:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoint_paths)} checkpoints to process")
    print()

    # Load neuron groups
    print("Loading neuron groups...")
    neuron_groups = load_neuron_groups(args.neuron_groups_dir)
    print()

    # Load initial weights
    print(f"Loading initial weights from {args.initial_weights_path}...")
    if not os.path.exists(args.initial_weights_path):
        print(f"ERROR: Initial weights not found at {args.initial_weights_path}")
        print("You need to run finetuning first to generate initial weights.")
        return

    initial_weights = torch.load(args.initial_weights_path)
    print(f"Loaded {len(initial_weights)} initial weight values")
    print()

    # Create output directory
    os.makedirs(args.drift_log_dir, exist_ok=True)

    # Process each checkpoint
    for checkpoint_path in tqdm(checkpoint_paths, desc="Processing checkpoints"):
        step = extract_step_from_checkpoint(checkpoint_path)

        # Check if drift log already exists
        drift_log_path = os.path.join(args.drift_log_dir, f"drift_step_{step:06d}.json")
        if os.path.exists(drift_log_path):
            print(f"\nStep {step}: Drift log already exists, skipping")
            continue

        print(f"\nProcessing checkpoint at step {step}...")
        print(f"  Checkpoint: {checkpoint_path}")

        # Load checkpoint with LoRA merge
        print("  Loading model...")
        model = load_checkpoint(
            args.base_model_path,
            checkpoint_path,
            device="auto"
        )

        # Compute drift
        print("  Computing drift metrics...")
        drift_metrics = compute_drift_metrics(
            model,
            initial_weights,
            neuron_groups,
            device="cuda"
        )

        # Save drift log
        save_drift_log(drift_metrics, step, args.drift_log_dir)
        print(f"  Saved drift log to {drift_log_path}")

        # Print summary for this checkpoint
        print(f"  Drift summary:")
        for metric_name, metric_data in drift_metrics.items():
            for group_name, stats in metric_data.items():
                print(f"    {group_name} {metric_name}: mean={stats['mean']:.6f}")

        # Clean up to free GPU memory
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Drift Computation Complete!")
    print("=" * 70)
    print(f"Drift logs saved to: {args.drift_log_dir}")
    print(f"Total checkpoints processed: {len(checkpoint_paths)}")


if __name__ == "__main__":
    main()
