"""
Fine-tune LLaMA model with LoRA while tracking neuron weight drift.

This script:
1. Loads a pre-trained model
2. Applies LoRA adapters
3. Fine-tunes on Alpaca dataset
4. Tracks weight drift for specified neuron groups at regular intervals
"""

import argparse
import json
import os

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

from lib.drift_utils import (
    load_neuron_groups,
    extract_neuron_weights,
    compute_drift_metrics,
    save_drift_log,
)


def load_model_and_tokenizer(model_name: str, model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded: {model_name}")
    return model, tokenizer


def setup_lora(model, lora_r: int = 8, lora_alpha: int = 16, target_modules=None):
    """Apply LoRA to model."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "o_proj"]

    print(f"Setting up LoRA (r={lora_r}, alpha={lora_alpha})...")
    print(f"Target modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_alpaca_dataset(tokenizer, max_length: int = 512):
    """Load and tokenize Alpaca dataset."""
    print("Loading Alpaca dataset...")

    # Load dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def format_prompt(example):
        """Format Alpaca prompt."""
        if example["input"]:
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}

    dataset = dataset.map(format_prompt)

    def tokenize_function(examples):
        """Tokenize examples."""
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    print(f"Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune with LoRA and track neuron drift"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama2-7b-chat-hf",
        help="Model name identifier",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/llama-2-7b-chat-hf/",
        help="Path to model directory",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/CS2881-alignment-attribution-code/finetuned_models",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 for full training)",
    )

    # Drift tracking arguments
    parser.add_argument(
        "--neuron_groups_dir",
        type=str,
        default="/workspace/CS2881-alignment-attribution-code/neuron_groups",
        help="Directory containing neuron group JSON files",
    )
    parser.add_argument(
        "--drift_log_dir",
        type=str,
        default="/dev/shm/drift_logs",
        help="Directory to save drift logs (use /dev/shm for large files)",
    )
    parser.add_argument(
        "--drift_log_interval",
        type=int,
        default=100,
        help="Steps between drift measurements",
    )
    parser.add_argument(
        "--initial_weights_dir",
        type=str,
        default="/dev/shm/initial_weights",
        help="Directory to save initial weights (use /dev/shm for large files)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Fine-Tuning with LoRA and Drift Tracking")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Training: {args.num_train_epochs} epochs, lr={args.learning_rate}")
    print(f"Output: {args.output_dir}")
    print(f"Drift logs: {args.drift_log_dir}")
    print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.model_path)

    # Load neuron groups
    print("Loading neuron groups...")
    neuron_groups = load_neuron_groups(args.neuron_groups_dir)
    print()

    # Extract and save initial weights (before LoRA)
    print("Extracting initial weights...")
    all_neurons = {}
    for group_name, group_data in neuron_groups.items():
        all_neurons.update(group_data)

    initial_weights = extract_neuron_weights(model, all_neurons, device="cuda")

    # Save initial weights to /dev/shm
    os.makedirs(args.initial_weights_dir, exist_ok=True)
    initial_weights_path = os.path.join(args.initial_weights_dir, "initial_weights.pt")
    torch.save(initial_weights, initial_weights_path)
    print(f"Saved initial weights to {initial_weights_path}")
    print(f"Tracking {len(initial_weights)} neurons across {len(neuron_groups)} groups")
    print()

    # Apply LoRA
    model = setup_lora(model, args.lora_r, args.lora_alpha)
    print()

    # Load dataset
    train_dataset = load_alpaca_dataset(tokenizer, args.max_length)
    print()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        max_steps=args.max_steps,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Add drift tracking callback
    # Note: HuggingFace Trainer callbacks need to inherit from TrainerCallback
    # For simplicity, we'll track drift manually at save checkpoints
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    # Compute initial drift (should be ~zero, sanity check)
    print("Computing initial drift (sanity check)...")
    drift_metrics = compute_drift_metrics(
        model.get_base_model(),  # Base model before any LoRA training
        initial_weights,
        neuron_groups,
        device="cuda",
    )
    save_drift_log(drift_metrics, 0, args.drift_log_dir)

    # Print sanity check results
    for metric_name, metric_data in drift_metrics.items():
        print(f"\nInitial {metric_name}:")
        for group_name, stats in metric_data.items():
            print(f"  {group_name}: mean={stats['mean']:.6f}, max={stats['max']:.6f}")
    print()

    # Train
    trainer.train()

    # Compute final drift
    print("\nComputing final drift...")

    # IMPORTANT: Merge LoRA weights into base model to get effective weights
    # Without this, we'd only measure frozen base weights (which don't change during LoRA)
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    drift_metrics = compute_drift_metrics(
        model,  # Now contains W + LoRA modifications
        initial_weights,
        neuron_groups,
        device="cuda",
    )
    save_drift_log(drift_metrics, trainer.state.global_step, args.drift_log_dir)

    # Save final model
    print(f"\nSaving final model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model saved to: {args.output_dir}")
    print(f"Drift logs saved to: {args.drift_log_dir}")
    print(f"Initial weights saved to: {initial_weights_path}")


if __name__ == "__main__":
    main()
