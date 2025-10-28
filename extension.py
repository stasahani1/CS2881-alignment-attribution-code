"""
Extension for Safety-Critical Neuron Analysis and Fine-tuning

This module implements a pipeline for:
1. Identifying safety-critical neurons using SNIP/Wanda scores
2. Freezing these neurons and fine-tuning the model
3. Recalculating SNIP/Wanda scores on the fine-tuned model

Based on the alignment attribution research framework.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import json
from typing import Dict, List, Optional, Tuple
import copy
from main import modeltype2path, get_llm

# Import with error handling for optional dependencies
try:
    from lib.eval import eval_ppl, eval_zero_shot, eval_attack
    from lib.prune import check_sparsity, prune_wanda, get_mask, find_layers
    from lib.model_wrapper import prune_wandg, make_Act, revert_Act_to_Linear, ActLinear, no_act_recording
    from lib.data import get_loaders
    LIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some lib modules not available: {e}")
    print("This is expected in test environments. The extension will work with full dependencies.")
    LIB_AVAILABLE = False

class SafetyNeuronAnalyzer:
    """
    Analyzes and manages safety-critical neurons in language models.
    """
    
    def __init__(self, model, tokenizer, device=torch.device("cuda:0")):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.safety_critical_neurons = {}
        self.original_weights = {}
        
    def identify_safety_critical_neurons(self, 
                                      prune_method: str = "wandg", 
                                      prune_data: str = "align_short",
                                      sparsity_ratio: float = 0.1,
                                      nsamples: int = 128,
                                      seed: int = 0) -> Dict[str, torch.Tensor]:
        """
        Identify safety-critical neurons using SNIP/Wanda scores WITHOUT pruning.
        This approach calculates importance scores and selects top-k% without modifying the model.
        
        Args:
            prune_method: Method to use ("wandg" for SNIP, "wanda" for Wanda)
            prune_data: Dataset to use for safety evaluation ("align" or "align_short")
            sparsity_ratio: Fraction of neurons to identify as safety-critical
            nsamples: Number of samples to use for scoring
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping layer names to boolean masks indicating safety-critical neurons
        """
        if not LIB_AVAILABLE:
            raise ImportError("Required lib modules not available. Please install all dependencies.")
        
        print(f"Calculating {prune_method} scores on {prune_data} dataset...")
        
        # Convert model to ActLinear for score calculation (non-destructive)
        model = make_Act(self.model, verbose=False)
        
        # Load calibration data
        dataloader, _ = get_loaders(
            prune_data,
            nsamples=nsamples,
            seed=seed,
            seqlen=model.seqlen,
            tokenizer=self.tokenizer,
            disentangle=True
        )
        print("Dataset loading complete")
        
        safety_masks = {}
        
        # Process each layer to calculate importance scores
        num_hidden_layers = model.config.num_hidden_layers
        for layer in range(num_hidden_layers):
            layer_filter_fn = lambda x: f"layers.{layer}." in x
            
            print(f"Processing layer {layer}...")
            
            # Enable gradients for this layer only
            model.zero_grad()
            model.requires_grad_(False)
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    print(f"enabling grad for {name}")
                    module.base.requires_grad_(True)
                    module.base.zero_grad()
            
            # Calculate gradients (importance scores)
            for batch in dataloader:
                inp, tar = batch[0].to(self.device), batch[1].to(self.device)
                model.zero_grad()
                with no_act_recording(model):
                    loss = model(input_ids=inp, labels=tar)[0]
                loss.backward()
            
            # Extract scores and create masks for this layer
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    if prune_method == "wandg":
                        # SNIP: Use gradient magnitude as importance score
                        scores = torch.abs(module.base.weight.grad)
                    elif prune_method == "wanda":
                        # Wanda: Use |weight| * sqrt(activation_norm)
                        scores = torch.abs(module.base.weight.data) * torch.sqrt(
                            module.activation_norms.reshape((1, -1))
                        )
                    else:
                        raise ValueError(f"Unsupported prune method: {prune_method}")
                    
                    # Select top-k% as safety-critical neurons
                    flat_scores = scores.flatten()
                    num_to_select = int(flat_scores.numel() * sparsity_ratio)
                    
                    if num_to_select > 0:
                        # Get threshold for top-k%
                        threshold_idx = flat_scores.numel() - num_to_select
                        threshold = torch.topk(flat_scores, threshold_idx, largest=False)[0][-1]
                        
                        # Create mask for safety-critical neurons (top-k%)
                        mask = scores >= threshold
                    else:
                        # If sparsity_ratio is 0, no neurons are safety-critical
                        mask = torch.zeros_like(scores, dtype=torch.bool)
                    
                    safety_masks[name] = mask
        
        # Convert back to regular model (non-destructive)
        model = revert_Act_to_Linear(model)
        
        print(f"Identified safety-critical neurons in {len(safety_masks)} layers")
        total_critical = sum(mask.sum().item() for mask in safety_masks.values())
        total_neurons = sum(mask.numel() for mask in safety_masks.values())
        print(f"Total safety-critical neurons: {total_critical}/{total_neurons} ({100*total_critical/total_neurons:.2f}%)")
        
        # Use existing check_sparsity function for additional analysis
        if LIB_AVAILABLE:
            print("Checking model sparsity...")
            sparsity = check_sparsity(self.model)
            print(f"Overall model sparsity: {sparsity:.4f}")
        
        self.safety_critical_neurons = safety_masks
        return safety_masks
    
    def freeze_safety_critical_neurons(self, model: nn.Module) -> nn.Module:
        """
        Freeze safety-critical neurons by setting their gradients to zero.
        
        Args:
            model: The model to freeze neurons in
            
        Returns:
            The model with frozen safety-critical neurons
        """
        print("Freezing safety-critical neurons...")
        
        frozen_count = 0
        total_count = 0
        
        # Use existing find_layers function for consistency
        linear_layers = find_layers(model)
        for name, module in linear_layers.items():
            if name in self.safety_critical_neurons:
                mask = self.safety_critical_neurons[name]
                
                # Create a hook to zero out gradients for safety-critical neurons
                def create_gradient_hook(mask):
                    def hook(grad):
                        # Zero out gradients for safety-critical neurons
                        grad[mask] = 0
                        return grad
                    return hook
                
                # Register the hook
                if module.weight.requires_grad:
                    module.weight.register_hook(create_gradient_hook(mask))
                
                frozen_count += mask.sum().item()
                total_count += mask.numel()
        
        print(f"Frozen {frozen_count}/{total_count} safety-critical neurons ({100*frozen_count/total_count:.2f}%)")
        return model
    


class FineTuner:
    """
    Handles fine-tuning of models with frozen safety-critical neurons.
    """
    
    def __init__(self, model, tokenizer, device=torch.device("cuda:0")):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def fine_tune_model(self, 
                       training_data: str = "alpaca_cleaned_no_safety",
                       num_epochs: int = 3,
                       learning_rate: float = 2e-5,
                       batch_size: int = 4,
                       max_length: int = 512,
                       save_path: str = "./fine_tuned_model") -> nn.Module:
        """
        Fine-tune the model while keeping safety-critical neurons frozen.
        
        Args:
            training_data: Dataset to use for fine-tuning
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            max_length: Maximum sequence length
            save_path: Path to save the fine-tuned model
            
        Returns:
            The fine-tuned model
        """
        if not LIB_AVAILABLE:
            raise ImportError("Required lib modules not available. Please install all dependencies.")
        
        print(f"Fine-tuning model on {training_data} dataset...")
        
        # Load training data
        trainloader, _ = get_loaders(
            training_data,
            nsamples=1000,  # Use more samples for actual fine-tuning
            seed=42,
            seqlen=max_length,
            tokenizer=self.tokenizer,
            disentangle=True
        )
        
        # Convert to HuggingFace dataset format
        def tokenize_function(examples):
            # This is a simplified tokenization - in practice you'd want more sophisticated handling
            inputs = []
            targets = []
            for inp, tar in zip(examples['input_ids'], examples['labels']):
                inputs.append(inp)
                targets.append(tar)
            return {'input_ids': inputs, 'labels': targets}
        
        # Create dataset
        dataset_dict = {
            'input_ids': [inp.squeeze() for inp, _ in trainloader],
            'labels': [tar.squeeze() for _, tar in trainloader]
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Fine-tuning completed. Model saved to {save_path}")
        return self.model




def main():
    """
    Main pipeline for safety-critical neuron analysis and fine-tuning.
    """
    parser = argparse.ArgumentParser(description="Safety-Critical Neuron Analysis and Fine-tuning")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='llama2-7b-chat-hf', help='Model name to analyze')
    parser.add_argument('--cache_dir', type=str, default='llm_weights', help='Cache directory for models')
    
    # Safety analysis arguments
    parser.add_argument('--prune_method', type=str, default='wandg', choices=['wandg', 'wanda'], 
                       help='Method for identifying safety-critical neurons')
    parser.add_argument('--prune_data', type=str, default='align', 
                       choices=['align', 'align_short'], help='Dataset for safety analysis')
    parser.add_argument('--sparsity_ratio', type=float, default=0.1, 
                       help='Fraction of neurons to identify as safety-critical')
    parser.add_argument('--nsamples', type=int, default=128, 
                       help='Number of samples for safety analysis')
    
    # Fine-tuning arguments
    parser.add_argument('--training_data', type=str, default='alpaca_cleaned_no_safety',
                       help='Dataset for fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, default='./fine_tuned_model', 
                       help='Path to save fine-tuned model')
    parser.add_argument('--results_path', type=str, default='./results', 
                       help='Path to save analysis results')
    
    # Evaluation arguments (following main.py pattern)
    parser.add_argument('--eval_ppl', action='store_true', help='Evaluate perplexity')
    parser.add_argument('--eval_zero_shot', action='store_true', help='Evaluate zero-shot performance')
    parser.add_argument('--eval_attack', action='store_true', help='Evaluate attack success rate')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # Memory management: Enable memory efficient attention and other optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "CPU mode")
    
    model = get_llm(args.model, args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(modeltype2path[args.model], use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Model loaded")
    
    # Step 1: Identify safety-critical neurons
    print("\n" + "="*50)
    print("STEP 1: Identifying safety-critical neurons")
    print("="*50)
    
    analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
    safety_masks = analyzer.identify_safety_critical_neurons(
        prune_method=args.prune_method,
        prune_data=args.prune_data,
        sparsity_ratio=args.sparsity_ratio,
        nsamples=args.nsamples,
        seed=args.seed
    )
    
    # Clear memory after Step 1
    torch.cuda.empty_cache()
    print(f"GPU memory after Step 1: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Step 1 complete")
    
    # Step 2: Freeze safety-critical neurons and fine-tune
    print("\n" + "="*50)
    print("STEP 2: Freezing neurons and fine-tuning")
    print("="*50)
    
    # Freeze safety-critical neurons
    model = analyzer.freeze_safety_critical_neurons(model)
    
    # Fine-tune the model
    fine_tuner = FineTuner(model, tokenizer, device)
    fine_tuned_model = fine_tuner.fine_tune_model(
        training_data=args.training_data,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_path=args.save_path
    )
    
    # Clear memory after Step 2
    torch.cuda.empty_cache()
    print(f"GPU memory after Step 2: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Step 2 complete")
    
    # Step 3: Recalculate safety-critical neuron scores
    print("\n" + "="*50)
    print("STEP 3: Recalculating safety-critical neuron scores")
    print("="*50)
    
    # Store original safety masks before updating analyzer
    original_masks = analyzer.safety_critical_neurons.copy()
    
    # Update analyzer to use the fine-tuned model
    analyzer.model = fine_tuned_model
    
    # Reuse the same method to identify safety-critical neurons on fine-tuned model
    new_safety_masks = analyzer.identify_safety_critical_neurons(
        prune_method=args.prune_method,
        prune_data=args.prune_data,
        sparsity_ratio=args.sparsity_ratio,
        nsamples=args.nsamples,
        seed=args.seed
    )
    
    # Compare with original safety-critical neurons
    print("\nComparing safety-critical neurons before and after fine-tuning:")
    for layer_name in new_safety_masks:
        if layer_name in original_masks:
            original_mask = original_masks[layer_name]
            new_mask = new_safety_masks[layer_name]
            
            # Calculate overlap
            overlap = (original_mask & new_mask).sum().item()
            original_count = original_mask.sum().item()
            new_count = new_mask.sum().item()
            
            print(f"Layer {layer_name}:")
            print(f"  Original safety-critical: {original_count}")
            print(f"  New safety-critical: {new_count}")
            print(f"  Overlap: {overlap} ({100*overlap/max(original_count, new_count, 1):.1f}%)")
    
    # Clear memory after Step 3
    torch.cuda.empty_cache()
    print(f"GPU memory after Step 3: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Step 3 complete")
    
    # Step 4: Evaluate models (optional, using existing eval functions)
    evaluation_results = {}
    if args.eval_ppl or args.eval_zero_shot or args.eval_attack:
        print("\n" + "="*50)
        print("STEP 4: Model Evaluation")
        print("="*50)
        
        # Evaluate original model
        print("Evaluating original model...")
        original_results = {}
        if args.eval_ppl:
            original_results['ppl'] = eval_ppl(model, tokenizer, "wikitext")
        if args.eval_zero_shot:
            original_results['zero_shot'] = eval_zero_shot(
                modeltype2path[args.model], model, tokenizer
            )
        if args.eval_attack:
            original_results['attack'] = eval_attack(model, tokenizer)
        
        # Evaluate fine-tuned model
        print("Evaluating fine-tuned model...")
        finetuned_results = {}
        if args.eval_ppl:
            finetuned_results['ppl'] = eval_ppl(fine_tuned_model, tokenizer, "wikitext")
        if args.eval_zero_shot:
            finetuned_results['zero_shot'] = eval_zero_shot(
                modeltype2path[args.model], fine_tuned_model, tokenizer
            )
        if args.eval_attack:
            finetuned_results['attack'] = eval_attack(fine_tuned_model, tokenizer)
        
        evaluation_results = {
            'original_model': original_results,
            'fine_tuned_model': finetuned_results
        }
        
        print("Evaluation results:")
        print(f"Original model: {original_results}")
        print(f"Fine-tuned model: {finetuned_results}")
    
    # Save results
    print("\n" + "="*50)
    print("Saving results")
    print("="*50)
    
    os.makedirs(args.results_path, exist_ok=True)
    
    # Save safety masks
    torch.save(safety_masks, os.path.join(args.results_path, 'original_safety_masks.pt'))
    torch.save(new_safety_masks, os.path.join(args.results_path, 'new_safety_masks.pt'))
    
    # Save analysis summary
    summary = {
        'original_safety_critical_count': sum(mask.sum().item() for mask in safety_masks.values()),
        'new_safety_critical_count': sum(mask.sum().item() for mask in new_safety_masks.values()),
        'total_neurons': sum(mask.numel() for mask in safety_masks.values()),
        'prune_method': args.prune_method,
        'prune_data': args.prune_data,
        'sparsity_ratio': args.sparsity_ratio,
        'training_data': args.training_data,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'memory_efficient': True,  # Flag indicating memory optimizations were used
        'evaluation_results': evaluation_results
    }
    
    with open(os.path.join(args.results_path, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete! Results saved to {args.results_path}")
    print(f"Fine-tuned model saved to {args.save_path}")
    print(f"Final GPU memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Analysis complete")


if __name__ == '__main__':
    main()
