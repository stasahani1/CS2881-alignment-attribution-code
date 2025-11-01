"""
Compute final drift from a trained checkpoint.

Use this if training completed but final drift wasn't saved.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from lib.drift_utils import (
    load_neuron_groups,
    compute_drift_metrics,
    save_drift_log,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute final drift from checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to final checkpoint directory",
    )
    parser.add_argument(
        "--neuron_groups_dir",
        type=str,
        default="/workspace/CS2881-alignment-attribution-code/neuron_groups",
        help="Directory containing neuron group JSON files",
    )
    parser.add_argument(
        "--initial_weights_path",
        type=str,
        default="/dev/shm/initial_weights/initial_weights.pt",
        help="Path to initial weights file",
    )
    parser.add_argument(
        "--drift_log_dir",
        type=str,
        default="/dev/shm/drift_logs",
        help="Directory to save drift logs",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step number (extracted from checkpoint path if not provided)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Computing Final Drift from Checkpoint")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Initial weights: {args.initial_weights_path}")
    print(f"Neuron groups: {args.neuron_groups_dir}")
    print()

    # Extract step number from checkpoint path if not provided
    if args.step is None:
        # Try to extract from path like "checkpoint-1500"
        basename = os.path.basename(args.checkpoint_path.rstrip("/"))
        if "checkpoint-" in basename:
            args.step = int(basename.split("-")[1])
        else:
            args.step = 9999  # Default if can't parse

    print(f"Step number: {args.step}")
    print()

    # Load neuron groups
    print("Loading neuron groups...")
    neuron_groups = load_neuron_groups(args.neuron_groups_dir)
    print()

    # Load initial weights
    print("Loading initial weights...")
    if not os.path.exists(args.initial_weights_path):
        print(f"ERROR: Initial weights not found at {args.initial_weights_path}")
        print("You need to run the finetuning script first to generate initial weights.")
        return

    initial_weights = torch.load(args.initial_weights_path)
    print(f"Loaded {len(initial_weights)} initial weights")
    print()

    # Load model from checkpoint
    # IMPORTANT: LoRA checkpoints only contain adapters, not the full model
    # We need to: 1) load base model, 2) load LoRA adapters, 3) merge them
    print("Loading base model and LoRA adapters from checkpoint...")

    # Load LoRA config to get base model path
    peft_config = PeftConfig.from_pretrained(args.checkpoint_path)
    base_model_path = peft_config.base_model_name_or_path

    print(f"  Base model: {base_model_path}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapters on top of base model
    print("  Loading LoRA adapters...")
    model_with_lora = PeftModel.from_pretrained(base_model, args.checkpoint_path)

    # Merge LoRA weights into base model
    print("  Merging LoRA adapters into base model...")
    model = model_with_lora.merge_and_unload()

    print("Model loaded and merged")
    print()

    # Compute drift
    print("Computing drift metrics...")
    drift_metrics = compute_drift_metrics(
        model,
        initial_weights,
        neuron_groups,
        device="cuda",
    )
    print()

    # Save drift log
    print(f"Saving drift log for step {args.step}...")
    save_drift_log(drift_metrics, args.step, args.drift_log_dir)
    print()

    # Print summary
    print("=" * 70)
    print("Drift Metrics Summary")
    print("=" * 70)
    for metric_name, metric_data in drift_metrics.items():
        print(f"\n{metric_name.upper()}:")
        for group_name, stats in metric_data.items():
            print(f"  {group_name}:")
            print(f"    mean={stats['mean']:.6f}, std={stats['std']:.6f}")
            print(f"    median={stats['median']:.6f}")
            print(f"    range=[{stats['min']:.6f}, {stats['max']:.6f}]")

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)
    print(f"Drift log saved to: {args.drift_log_dir}/drift_step_{args.step:06d}.json")


if __name__ == "__main__":
    main()
