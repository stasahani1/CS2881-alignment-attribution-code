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


def freeze_safety_critical_neurons(model, neuron_groups_file: str):
    """
    Freeze safety-critical neurons by preventing LoRA adapters from updating them.
    
    For LoRA, base model weights are already frozen. To prevent LoRA adapters from
    affecting specific neurons, we:
    1. Find the LoRA adapter for each module
    2. Zero out corresponding rows in lora_B matrices (preventing LoRA updates to those output neurons)
    3. Set requires_grad=False and register hooks to prevent gradient updates
    
    Args:
        model: Model with LoRA applied (PEFT model)
        neuron_groups_file: Path to neuron_groups_set_diff.json file
    
    Returns:
        Number of neurons frozen
    """
    print(f"Loading safety-critical neurons from {neuron_groups_file}...")
    
    # Load the safety_set_diff neuron group
    if not os.path.exists(neuron_groups_file):
        raise FileNotFoundError(
            f"Neuron groups file not found: {neuron_groups_file}"
        )
    
    with open(neuron_groups_file, "r") as f:
        safety_neurons = json.load(f)
    
    # Get base model (LoRA wraps the base model)
    # PEFT models have a base_model attribute
    # Need to access the transformer model (LlamaModel) which has .layers
    if hasattr(model, 'base_model'):
        base_model_full = model.base_model.model
        # Check if we need to go one level deeper to get LlamaModel with .layers
        if hasattr(base_model_full, 'model') and hasattr(base_model_full.model, 'layers'):
            base_model = base_model_full.model  # This is LlamaModel with .layers
        else:
            base_model = base_model_full
    elif hasattr(model, 'get_base_model'):
        base_model_full = model.get_base_model()
        if hasattr(base_model_full, 'model') and hasattr(base_model_full.model, 'layers'):
            base_model = base_model_full.model
        else:
            base_model = base_model_full
    else:
        # Fallback: assume model is already base model or try direct access
        base_model = model.model if hasattr(model, 'model') else model
    
    frozen_count = 0
    hook_handles = []
    
    # Store original weight values for safety
    original_weights = {}
    
    # Track frozen rows per layer/module for LoRA adapter freezing
    frozen_rows_by_module = {}
    
    for layer_name, neuron_coords in safety_neurons.items():
        # Parse layer name: "layer_0_mlp.down_proj" -> layer 0, module mlp.down_proj
        parts = layer_name.split("_", 2)
        if len(parts) >= 3 and parts[0] == "layer":
            layer_idx = int(parts[1])
            module_path = parts[2]
            
            try:
                # Navigate to the layer
                layer = base_model.layers[layer_idx]
                
                # Navigate to the specific module (e.g., mlp.down_proj)
                module = layer
                for attr in module_path.split("."):
                    module = getattr(module, attr)
                
                # Get weight matrix
                weight = module.weight
                
                # Store neuron coordinates and track frozen rows
                valid_coords = []
                frozen_rows = set()
                
                # Batch process all neuron coordinates for better performance
                # Filter valid coordinates first
                for coord in neuron_coords:
                    if len(coord) == 2:
                        row, col = coord
                        # Bounds checking
                        if row < weight.shape[0] and col < weight.shape[1]:
                            valid_coords.append((row, col))
                            frozen_rows.add(row)  # Track which rows need to be frozen
                            frozen_count += 1
                        else:
                            print(f"Warning: ({row}, {col}) out of bounds for {layer_name} shape {weight.shape}, skipping")
                
                # Batch extract all weights at once (much faster than individual operations)
                if valid_coords:
                    # Convert to tensors for indexing
                    rows_tensor = torch.tensor([r for r, c in valid_coords], dtype=torch.long, device=weight.device)
                    cols_tensor = torch.tensor([c for r, c in valid_coords], dtype=torch.long, device=weight.device)
                    
                    # Extract all weights in one vectorized operation
                    weight_values = weight.data[rows_tensor, cols_tensor].float().cpu()
                    
                    # Store in original_weights dict
                    for idx, (row, col) in enumerate(valid_coords):
                        key = f"{layer_name}_{row}_{col}"
                        original_weights[key] = weight_values[idx]
                
                # Store frozen rows for this module to freeze in LoRA adapters
                if frozen_rows:
                    frozen_rows_by_module[layer_name] = {
                        'layer_idx': layer_idx,
                        'module_path': module_path,
                        'frozen_rows': frozen_rows,
                        'weight_shape': weight.shape
                    }
                
                print(f"Identified {len(valid_coords)} neurons to freeze in {layer_name} (affecting {len(frozen_rows)} rows)")
                    
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not access {layer_name}: {e}")
                continue
    
    # Now freeze LoRA adapters to prevent updates to frozen neurons
    print(f"\nFreezing LoRA adapters to prevent updates to frozen neurons...")
    
    # Access LoRA adapters through PEFT model structure
    # PEFT stores LoRA adapters as parameters with names like:
    # "base_model.model.layers.{i}.{module}.lora_A.weight" and "lora_B.weight"
    lora_adapter_count = 0
    
    # Build a mapping of layer/module to LoRA adapter parameters
    lora_adapters = {}
    print("\nDebug: Scanning for LoRA adapters...")
    lora_param_count = 0
    sample_names = []
    
    for name, param in model.named_parameters():
        if 'lora_B' in name or 'lora_A' in name:
            lora_param_count += 1
            if len(sample_names) < 5:  # Collect first 5 for debugging
                sample_names.append((name, param.shape))
            
            # Parse the name to extract layer and module info
            # Pattern: base_model.model.model.layers.{i}.{module}.lora_{A/B}.default.weight
            parts = name.split('.')
            try:
                # Find the layer index
                if 'layers' in parts:
                    layer_idx_pos = parts.index('layers')
                    if layer_idx_pos + 1 < len(parts):
                        layer_idx = int(parts[layer_idx_pos + 1])
                        # Find where lora_A or lora_B starts
                        # Structure: ...layers.{i}.{module_path}.lora_{A/B}.default.weight
                        lora_pos = None
                        for i in range(layer_idx_pos + 2, len(parts)):
                            if parts[i] in ['lora_A', 'lora_B']:
                                lora_pos = i
                                break
                        
                        if lora_pos is not None:
                            # Module path is everything between layers.{i} and lora_{A/B}
                            module_parts = parts[layer_idx_pos + 2:lora_pos]
                            module_path = '.'.join(module_parts)
                            
                            lora_type = 'lora_A' if 'lora_A' in name else 'lora_B'
                            key = (layer_idx, module_path)
                            
                            if key not in lora_adapters:
                                lora_adapters[key] = {}
                            lora_adapters[key][lora_type] = param
            except (ValueError, IndexError) as e:
                # Only print first few parsing errors to avoid spam
                if len(sample_names) <= 5:
                    print(f"  Warning: Failed to parse LoRA param '{name}': {e}")
                continue
    
    print(f"  Total LoRA parameters found: {lora_param_count}")
    print(f"  LoRA adapters mapped: {len(lora_adapters)}")
    if sample_names:
        print(f"  Sample LoRA parameter names:")
        for name, shape in sample_names[:3]:
            print(f"    {name}, shape: {shape}")
    if lora_adapters:
        print(f"  Sample adapter keys (layer_idx, module_path): {list(lora_adapters.keys())[:3]}")
    
    # Now freeze the LoRA adapters for modules with frozen neurons
    for layer_name, module_info in frozen_rows_by_module.items():
        layer_idx = module_info['layer_idx']
        module_path = module_info['module_path']
        frozen_rows = module_info['frozen_rows']
        
        try:
            # Find the LoRA adapter for this layer/module
            key = (layer_idx, module_path)
            
            if key in lora_adapters and 'lora_B' in lora_adapters[key]:
                lora_B = lora_adapters[key]['lora_B']
                
                # Freeze rows in lora_B that correspond to frozen neurons
                # lora_B shape: [out_features, r] where out_features matches weight.shape[0]
                if lora_B.shape[0] == module_info['weight_shape'][0]:
                    # Filter valid rows (within bounds)
                    valid_frozen_rows = [row for row in frozen_rows if row < lora_B.shape[0]]
                    
                    if valid_frozen_rows:
                        # Vectorized zeroing: zero all frozen rows at once (much faster)
                        frozen_rows_tensor = torch.tensor(valid_frozen_rows, dtype=torch.long, device=lora_B.device)
                        lora_B.data[frozen_rows_tensor, :] = 0.0
                        
                        # Create a single hook function that handles all frozen rows via set lookup
                        # This is much more efficient than one hook per row
                        frozen_rows_set = set(valid_frozen_rows)
                        
                        def lora_freeze_hook(grad):
                            """Hook to zero gradients for all frozen LoRA rows."""
                            if grad is not None:
                                grad = grad.clone()
                                # Vectorized: zero all frozen rows at once
                                if len(frozen_rows_set) > 0:
                                    frozen_rows_tensor_hook = torch.tensor(list(frozen_rows_set), dtype=torch.long, device=grad.device)
                                    grad[frozen_rows_tensor_hook, :] = 0.0
                            return grad
                        
                        # Register single hook for all frozen rows
                        if lora_B.requires_grad:
                            handle = lora_B.register_hook(lora_freeze_hook)
                            hook_handles.append(handle)
                    
                    lora_adapter_count += 1
                    print(f"  Frozen LoRA adapter for {layer_name}: zeroed {len(valid_frozen_rows)} rows in lora_B")
                else:
                    print(f"  Warning: LoRA adapter shape mismatch for {layer_name}: expected {module_info['weight_shape'][0]}, got {lora_B.shape[0]}")
            else:
                # Check if this module is even targeted by LoRA (might not be in target_modules)
                print(f"  Info: No LoRA adapter found for {layer_name} (module may not be targeted by LoRA)")
                
        except (AttributeError, IndexError, KeyError) as e:
            print(f"  Warning: Could not freeze LoRA adapter for {layer_name}: {e}")
            continue
    
    print(f"\nTotal frozen neurons: {frozen_count}")
    print(f"Frozen LoRA adapters for {lora_adapter_count} modules")
    print(f"Registered {len(hook_handles)} gradient hooks")
    
    # Store hook handles in model for potential cleanup
    if not hasattr(model, '_frozen_neuron_hooks'):
        model._frozen_neuron_hooks = hook_handles
        model._frozen_neuron_weights = original_weights
    
    # Clean up local variables to free memory
    del safety_neurons
    del base_model
    del frozen_rows_by_module
    del lora_adapters
    
    # Clear GPU cache after freezing to free up memory
    torch.cuda.empty_cache()
    
    return frozen_count


def load_alpaca_dataset(tokenizer, max_length: int = 512):
    """Load and tokenize Alpaca dataset."""
    print("Loading Alpaca dataset...")

    # Load dataset from CSV file directly using datasets library
    dataset = load_dataset("csv", data_files="data/alpaca_cleaned_no_safety_train.csv", split="train")

    def format_prompt(example):
        """Format Alpaca prompt from CSV data."""
        # CSV has 'prompt' and 'response' columns
        # prompt already contains [INST] tags, so we just append response
        text = f"{example['prompt']} {example['response']}"
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

    # Clean up intermediate dataset to free memory
    del dataset

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
        "--initial_weights_dir",
        type=str,
        default="/dev/shm/initial_weights",
        help="Directory to save initial weights (needed for drift computation)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (deprecated: not used, only final model is saved)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., finetuned_models/checkpoint-1000)",
    )
    parser.add_argument(
        "--freeze_safety_neurons",
        type=str,
        default=None,
        help="Path to neuron_groups_set_diff.json file to freeze safety-critical neurons during training (e.g., outputs/neuron_groups/neuron_groups_set_diff.json)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Fine-Tuning with LoRA")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Training: {args.num_train_epochs} epochs, lr={args.learning_rate}")
    print(f"Output: {args.output_dir}")
    print(f"Note: Only final model will be saved (no intermediate checkpoints)")
    if args.resume_from_checkpoint:
        print(f"Resuming from: {args.resume_from_checkpoint}")
    if args.freeze_safety_neurons:
        print(f"Freezing safety neurons from: {args.freeze_safety_neurons}")
    print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.model_path)

    # Load neuron groups
    print("Loading neuron groups...")
    neuron_groups = load_neuron_groups(args.neuron_groups_dir)
    print()

    # Extract and save initial weights (before LoRA) - only if not freezing
    initial_weights_path = None
    if not args.freeze_safety_neurons:
        print("Extracting initial weights...")
        # Merge all neuron groups into a single dict, combining coords for overlapping layers
        all_neurons = {}
        for group_name, group_data in neuron_groups.items():
            for layer_name, coords in group_data.items():
                if layer_name not in all_neurons:
                    all_neurons[layer_name] = []
                all_neurons[layer_name].extend(coords)

        initial_weights = extract_neuron_weights(model, all_neurons, device="cuda")

        # Save initial weights to /dev/shm
        os.makedirs(args.initial_weights_dir, exist_ok=True)
        initial_weights_path = os.path.join(args.initial_weights_dir, "initial_weights.pt")
        torch.save(initial_weights, initial_weights_path)
        print(f"Saved initial weights to {initial_weights_path}")
        print(f"Tracking {len(initial_weights)} neurons across {len(neuron_groups)} groups")
        
        # Clear memory after extracting weights
        del initial_weights
        del all_neurons  # Clean up intermediate dictionary
        torch.cuda.empty_cache()
        print()
    else:
        print("Skipping initial weights extraction (freezing enabled)")
        print()

    # Apply LoRA
    model = setup_lora(model, args.lora_r, args.lora_alpha)
    
    torch.cuda.empty_cache()  # Clear cache after LoRA setup
    print()

    # Freeze safety-critical neurons if requested
    if args.freeze_safety_neurons:
        print("=" * 70)
        print("Freezing Safety-Critical Neurons")
        print("=" * 70)
        freeze_safety_critical_neurons(model, args.freeze_safety_neurons)
        print()

    # Clean up neuron_groups after we're done with it (no longer needed)
    del neuron_groups
    torch.cuda.empty_cache()

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
        save_strategy="no",  # Don't save checkpoints during training, only save final model
        max_steps=args.max_steps,
        bf16=True,
        # Disable gradient checkpointing when freezing neurons to avoid conflicts
        # Gradient checkpointing with LoRA works, but can cause issues when freezing specific neurons
        gradient_checkpointing=not args.freeze_safety_neurons,  # Disable when freezing
        remove_unused_columns=False,
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train (drift will be computed separately in Phase 2b, if initial weights were saved)
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    if not args.freeze_safety_neurons:
        print("Note: Drift computation will be done post-training using compute_drift.py")
    print()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Clear memory after training
    torch.cuda.empty_cache()

    # Save final model
    print(f"\nSaving final model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Cleanup frozen neuron hooks if they exist
    if hasattr(model, '_frozen_neuron_hooks'):
        print("Cleaning up frozen neuron hooks...")
        for handle in model._frozen_neuron_hooks:
            handle.remove()
        del model._frozen_neuron_hooks
        # Also delete original_weights stored on model
        if hasattr(model, '_frozen_neuron_weights'):
            del model._frozen_neuron_weights
        torch.cuda.empty_cache()
    
    # Clean up trainer and dataset to free memory
    del trainer
    del train_dataset
    torch.cuda.empty_cache()

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Model checkpoints saved to: {args.output_dir}")
    if initial_weights_path:
        print(f"Initial weights saved to: {initial_weights_path}")
        print()
        print("Next step: Run compute_drift.py to compute drift for all checkpoints")
    else:
        print()


if __name__ == "__main__":
    main()
