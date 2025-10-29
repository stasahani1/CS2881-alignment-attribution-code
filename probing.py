"""
Probing script for identifying safety-critical attention heads.

This script implements the probing method described in the paper:
1. Feed model with harmful and harmless instructions
2. Collect activation outputs from each attention head
3. Train linear classifiers to distinguish between harmful/harmless activations
4. Identify heads with highest accuracy as safety-critical
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_harmful_instructions(file_path, n_samples=420):
    """Load harmful instructions from advbench.txt."""
    with open(file_path, 'r') as f:
        instructions = [line.strip() for line in f if line.strip()]

    # Sample n_samples if we have more
    if len(instructions) > n_samples:
        random.shuffle(instructions)
        instructions = instructions[:n_samples]

    return instructions


def load_harmless_instructions(file_path, n_samples=420):
    """Load harmless instructions from utility dataset."""
    df = pd.read_csv(file_path)

    # Sample random harmless instructions
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)

    # Extract instructions from the dataset
    instructions = []
    for _, row in df.iterrows():
        if 'prompt' in df.columns:
            # Remove [INST] tags if present
            prompt = row['prompt']
            # Remove [INST] and [/INST] tags
            prompt = prompt.replace('[INST]', '').replace('[/INST]', '').strip()
            instructions.append(prompt)
        elif 'text' in df.columns:
            # Try to extract instruction from text
            text = row['text']
            # Simple heuristic: take the first sentence or first 100 chars
            instruction = text.split('\n')[0][:200]
            instructions.append(instruction)

    return instructions[:n_samples]


class InstructionDataset(Dataset):
    """Dataset for instructions with labels."""

    def __init__(self, instructions, labels, tokenizer, max_length=512):
        self.instructions = instructions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            instruction,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class AttentionHeadActivationCollector:
    """Collector for attention head activations.

    This collector hooks into the attention mechanism to extract per-head
    activations before they are combined through the output projection.
    """

    def __init__(self, model, num_layers, num_heads, head_dim):
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.activations = {f"layer-{l}-head-{h}": []
                           for l in range(num_layers)
                           for h in range(num_heads)}
        self.hooks = []
        self.temp_storage = {}

    def _get_o_proj_input_hook(self, layer_idx):
        """Hook to capture input to output projection (head outputs before combination)."""
        def hook(module, input, output):
            # The input to o_proj is the concatenated head outputs
            # Shape: (batch_size, seq_len, hidden_dim)
            if isinstance(input, tuple):
                head_outputs = input[0]
            else:
                head_outputs = input

            batch_size, seq_len, hidden_dim = head_outputs.shape

            # Mean pool over sequence length to get fixed-size representation
            # Shape: (batch_size, hidden_dim)
            pooled = head_outputs.mean(dim=1)

            # Reshape to separate heads: (batch_size, num_heads, head_dim)
            head_activations = pooled.view(batch_size, self.num_heads, self.head_dim)

            # Store each head's activation
            for h in range(self.num_heads):
                head_key = f"layer-{layer_idx}-head-{h}"
                # Shape: (batch_size, head_dim)
                head_act = head_activations[:, h, :].detach().cpu()
                self.activations[head_key].append(head_act)

        return hook

    def register_hooks(self):
        """Register forward hooks for output projection of all attention layers.

        For Llama models, we hook into the o_proj (output projection) module
        to capture the concatenated head outputs before they are combined.
        """
        for layer_idx in range(self.num_layers):
            # Hook into the output projection to get head outputs before combination
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_pre_hook(self._get_o_proj_input_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self):
        """Get collected activations as numpy arrays."""
        activations_np = {}
        for key, acts in self.activations.items():
            if len(acts) > 0:
                # Concatenate all batches
                activations_np[key] = torch.cat(acts, dim=0).numpy()
            else:
                print(f"Warning: No activations collected for {key}")
        return activations_np

    def clear_activations(self):
        """Clear stored activations."""
        for key in self.activations:
            self.activations[key] = []


def collect_activations(model, dataloader, collector, device):
    """Collect activations for all samples in dataloader."""
    model.eval()
    collector.clear_activations()
    collector.register_hooks()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting activations"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    collector.remove_hooks()
    activations = collector.get_activations()

    return activations


def train_linear_classifiers(X_train, y_train, X_val, y_val, head_keys):
    """Train a linear classifier for each attention head."""
    results = {}

    for head_key in tqdm(head_keys, desc="Training classifiers"):
        # Get activations for this head
        X_train_head = X_train[head_key]
        X_val_head = X_val[head_key]

        # Train logistic regression classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_head, y_train)

        # Evaluate on validation set
        val_accuracy = clf.score(X_val_head, y_val)

        results[head_key] = val_accuracy

    return results


def main():
    parser = argparse.ArgumentParser(description="Probing for safety-critical attention heads")
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf",
                       help="Model name")
    parser.add_argument("--model_path", type=str, default="models/llama-2-7b-chat-hf/",
                       help="Path to model")
    parser.add_argument("--harmful_data", type=str, default="data/advbench.txt",
                       help="Path to harmful instructions")
    parser.add_argument("--harmless_data", type=str, default="data/alpaca_cleaned_no_safety_train.csv",
                       help="Path to harmless instructions")
    parser.add_argument("--n_samples", type=int, default=420,
                       help="Number of samples for each class")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_file", type=str, default="data/probing_result_7b.json",
                       help="Output file for probing results")
    parser.add_argument("--train_split", type=float, default=5/7,
                       help="Training split ratio (default: 5/7 for 5:2 train:val)")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode with small sample (10 harmful + 10 harmless)")

    args = parser.parse_args()

    # Override n_samples in test mode
    if args.test_mode:
        print("="*80)
        print("TEST MODE: Using small sample (10 harmful + 10 harmless)")
        print("="*80)
        args.n_samples = 10
        args.batch_size = 2

    # Set seed
    set_seed(args.seed)

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Get model configuration
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_dim = model.config.hidden_size
    head_dim = hidden_dim // num_heads
    print(f"Model has {num_layers} layers and {num_heads} attention heads per layer")
    print(f"Hidden dimension: {hidden_dim}, Head dimension: {head_dim}")

    # Load instructions
    print(f"Loading {args.n_samples} harmful instructions...")
    harmful_instructions = load_harmful_instructions(args.harmful_data, args.n_samples)
    print(f"Loaded {len(harmful_instructions)} harmful instructions")

    print(f"Loading {args.n_samples} harmless instructions...")
    harmless_instructions = load_harmless_instructions(args.harmless_data, args.n_samples)
    print(f"Loaded {len(harmless_instructions)} harmless instructions")

    # Combine instructions and create labels
    # Label 1 for harmful, 0 for harmless
    all_instructions = harmful_instructions + harmless_instructions
    all_labels = [1] * len(harmful_instructions) + [0] * len(harmless_instructions)

    # Split into train and validation
    train_instructions, val_instructions, train_labels, val_labels = train_test_split(
        all_instructions, all_labels,
        train_size=args.train_split,
        random_state=args.seed,
        stratify=all_labels
    )

    print(f"Train set: {len(train_instructions)} samples")
    print(f"Validation set: {len(val_instructions)} samples")

    # Create datasets and dataloaders
    train_dataset = InstructionDataset(train_instructions, train_labels, tokenizer, args.max_length)
    val_dataset = InstructionDataset(val_instructions, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create activation collector
    device = next(model.parameters()).device
    collector = AttentionHeadActivationCollector(model, num_layers, num_heads, head_dim)

    # Collect training activations
    print("\nCollecting training activations...")
    train_activations = collect_activations(model, train_loader, collector, device)

    # Collect validation activations
    print("\nCollecting validation activations...")
    val_activations = collect_activations(model, val_loader, collector, device)

    # Get head keys
    head_keys = sorted(train_activations.keys())

    # Train linear classifiers for each head
    print(f"\nTraining linear classifiers for {len(head_keys)} attention heads...")
    train_labels_np = np.array(train_labels)
    val_labels_np = np.array(val_labels)

    results = train_linear_classifiers(
        train_activations, train_labels_np,
        val_activations, val_labels_np,
        head_keys
    )

    # Sort by accuracy
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # Print top 10 heads
    print("\n" + "="*80)
    print("Top 10 safety-critical attention heads:")
    print("="*80)
    for i, (head_key, accuracy) in enumerate(list(sorted_results.items())[:10], 1):
        print(f"{i:2d}. {head_key:20s}: {accuracy:.4f}")

    # Print bottom 10 heads
    print("\n" + "="*80)
    print("Bottom 10 attention heads:")
    print("="*80)
    for i, (head_key, accuracy) in enumerate(list(sorted_results.items())[-10:], 1):
        print(f"{i:2d}. {head_key:20s}: {accuracy:.4f}")

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f)

    print("Done!")

    # Print summary statistics
    accuracies = list(results.values())
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    print(f"Mean accuracy: {np.mean(accuracies):.4f}")
    print(f"Std accuracy:  {np.std(accuracies):.4f}")
    print(f"Min accuracy:  {np.min(accuracies):.4f}")
    print(f"Max accuracy:  {np.max(accuracies):.4f}")
    print(f"Median accuracy: {np.median(accuracies):.4f}")


if __name__ == "__main__":
    main()
