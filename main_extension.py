"""
Extension for Safety-Critical Neuron Analysis and Fine-tuning

This module implements a pipeline for:
1. Identifying safety-critical neurons using SNIP/Wanda scores
2. Freezing these neurons and fine-tuning the model
3. Recalculating SNIP/Wanda scores on the fine-tuned model

Based on the alignment attribution research framework.
"""

import os
import copy
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from datasets import Dataset
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from main import modeltype2path, get_llm
from lib.extension_utils import SafetyNeuronAnalyzer, FineTuner



def main():
    """
    Main pipeline for safety-critical neuron analysis and fine-tuning.
    """
    parser = argparse.ArgumentParser(description="Safety-Critical Neuron Analysis and Fine-tuning")

    # Task arguments
    parser.add_argument("--task", type=str, default="identify_safety_neurons", choices=["identify_safety_neurons", "fine_tune", "eval"], help="Task to perform, select value from ['identify_safety_neurons', 'fine_tune', 'eval']")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf", help="Model name to analyze")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for remote models")
    
    # Safety neuron identification arguments
    parser.add_argument("--prune_method", type=str, default="wandg", choices=["wandg", "wanda"], 
                       help="Method for identifying safety-critical neurons")
    parser.add_argument("--prune_data", type=str, default="align", 
                       choices=["align", "align_short"], help="Dataset for safety analysis")
    parser.add_argument("--sparsity_ratio", type=float, default=0.1, 
                       help="Fraction of neurons to identify as safety-critical")
    parser.add_argument("--nsamples", type=int, default=128, 
                       help="Number of samples for safety analysis")
    
    # Fine-tuning arguments
    parser.add_argument("--training_data", type=str, default="alpaca_cleaned_no_safety",
                       help="Dataset for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--safety_masks", type=str, default=None, help="Path to safety masks")
    
    # Evaluation arguments 
    parser.add_argument("--original_safety_masks_path", type=str, default=None, help="Path to original safety masks")
    parser.add_argument("--fine_tuned_safety_masks_path", type=str, default=None, help="Path to fine-tuned safety masks")
    parser.add_argument("--original_safety_scores_path", type=str, default=None, help="Path to original safety scores")
    parser.add_argument("--fine_tuned_safety_scores_path", type=str, default=None, help="Path to fine-tuned safety scores")
    
    # Output arguments
    parser.add_argument("--model_save_path", type=str, default="./fine_tuned_model", 
                       help="Path to save fine-tuned model")
    parser.add_argument("--results_path", type=str, default="./results", 
                       help="Path to save analysis results")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
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

    if args.task != "eval":
        # Load model and tokenizer
        print(f"Loading model {args.model}...")
        print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "CPU mode")
        
        model = get_llm(args.model)
        tokenizer = AutoTokenizer.from_pretrained(modeltype2path[args.model], use_fast=False)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Model loaded")
    

    # Step 1: Identify safety-critical neurons
    if args.task == "identify_safety_neurons":
        print("\n" + "="*50)
        print("STEP 1: Identifying safety-critical neurons")
        print("="*50)
        
        analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
        safety_masks, original_safety_scores = analyzer.identify_safety_critical_neurons(
            prune_method=args.prune_method,
            prune_data=args.prune_data,
            sparsity_ratio=args.sparsity_ratio,
            nsamples=args.nsamples,
            seed=args.seed
        )
        
        # Clear memory after Step 1
        torch.cuda.empty_cache()
        print(f"GPU memory after Step 1: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Step 1 complete")

        # Save safety masks and scores
        print("\nSaving safety masks and scores...")
        save_dir = os.path.join(args.results_path, "safety_neurons")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save masks and scores 
        torch.save(safety_masks, os.path.join(save_dir, "original_safety_masks.pt"))
        torch.save(original_safety_scores, os.path.join(save_dir, "original_safety_scores.pt")) 
        
        print(f"Saved safety neuron data to {save_dir}")
    

    # Step 2: Freeze safety-critical neurons and fine-tune
    elif args.task == "fine_tune":
        print("\n" + "="*50)
        print("STEP 2: Freezing neurons and fine-tuning")
        print("="*50)
        
        # Load safety masks
        safety_masks = torch.load(args.safety_masks) 
        
        # Fine-tune the model (freezing is handled internally by FineTuner)
        fine_tuner = FineTuner(model, tokenizer, device, safety_masks) 
        
        print("Fine-tuning model with frozen safety-critical neurons...")
        fine_tuned_model = fine_tuner.fine_tune_model(
            training_data=args.training_data,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_save_path=args.model_save_path
        )
        
        # Clear memory after fine-tuning
        torch.cuda.empty_cache()
        print(f"GPU memory after fine-tuning: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Fine-tuning complete")
        
        # Recalculate safety-critical neuron scores
        print("\nRecalculating safety-critical neuron scores for fine-tuned model...")
        
        # Create new analyzer for fine-tuned model
        analyzer = SafetyNeuronAnalyzer(fine_tuned_model, tokenizer, device)
        
        # Identify safety-critical neurons on fine-tuned model
        new_safety_masks, new_safety_scores = analyzer.identify_safety_critical_neurons(
            prune_method=args.prune_method,
            prune_data=args.prune_data,
            sparsity_ratio=args.sparsity_ratio,
            nsamples=args.nsamples,
            seed=args.seed
        )
        
        # Clear memory after recalculating safety-critical neuron scores
        torch.cuda.empty_cache()
        print(f"GPU memory after recalculating safety-critical neuron scores: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Recalculation complete")

        # Save safety masks and scores
        print("\nSaving safety masks and scores for fine-tuned model...")
        save_dir = os.path.join(args.results_path, "safety_neurons")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save masks and scores 
        torch.save(new_safety_masks, os.path.join(save_dir, "fine_tuned_safety_masks.pt"))
        torch.save(new_safety_scores, os.path.join(save_dir, "fine_tuned_safety_scores.pt"))
        
        print(f"Saved fine-tuned safety neuron data to {save_dir}")
    
    # Step 3: Evaluation 
    elif args.task == "eval":
        print("\n" + "="*50)
        print("STEP 3: Evaluation")
        print("="*50)
        
        # Load safety neuron data and initialize tracking variables
        original_safety_masks = torch.load(args.original_safety_masks_path)
        fine_tuned_safety_masks = torch.load(args.fine_tuned_safety_masks_path)
        original_safety_scores = torch.load(args.original_safety_scores_path)
        fine_tuned_safety_scores = torch.load(args.fine_tuned_safety_scores_path)
        
        comparison_results = {}
        total_overlap = total_original = total_new = 0
        overall_original_scores = []
        overall_new_scores = []
        
        print("\nComparing safety-critical neurons and scores before/after fine-tuning:")
        
        # Process each layer
        for layer_name in fine_tuned_safety_scores:
            # Get masks and calculate overlap statistics
            original_mask = original_safety_masks[layer_name]
            new_mask = fine_tuned_safety_masks[layer_name]
            
            overlap = torch.logical_and(original_mask, new_mask).sum().item()
            original_count = original_mask.sum().item()
            new_count = new_mask.sum().item()
            
            total_overlap += overlap
            total_original += original_count
            total_new += new_count
            
            overlap_pct_original = 100 * overlap / original_count if original_count > 0 else 0
            overlap_pct_new = 100 * overlap / new_count if new_count > 0 else 0
            
            # Get score statistics
            original_scores = original_safety_scores[layer_name]
            new_scores = fine_tuned_safety_scores[layer_name]
            
            orig_stats = {
                "mean": original_scores.mean().item(),
                "std": original_scores.std().item(),
                "min": original_scores.min().item(),
                "max": original_scores.max().item()
            }
            
            new_stats = {
                "mean": new_scores.mean().item(),
                "std": new_scores.std().item(),
                "min": new_scores.min().item(),
                "max": new_scores.max().item()
            }
            
            # Store results
            comparison_results[layer_name] = {
                "original_count": original_count,
                "new_count": new_count,
                "overlap": overlap,
                "overlap_pct_original": overlap_pct_original,
                "overlap_pct_new": overlap_pct_new,
                "original_score_stats": orig_stats,
                "new_score_stats": new_stats
            }
            
            # Print layer results
            print(f"\nLayer {layer_name}:")
            print(f"  Neuron counts - Original: {original_count}, New: {new_count}, Overlap: {overlap}")
            print(f"  Overlap % - Of original preserved: {overlap_pct_original:.2f}%, Of new from original: {overlap_pct_new:.2f}%")
            print(f"  Original scores - Mean: {orig_stats['mean']:.4f}, Std: {orig_stats['std']:.4f}, Min: {orig_stats['min']:.4f}, Max: {orig_stats['max']:.4f}")
            print(f"  New scores     - Mean: {new_stats['mean']:.4f}, Std: {new_stats['std']:.4f}, Min: {new_stats['min']:.4f}, Max: {new_stats['max']:.4f}")
            
            overall_original_scores.append(original_scores)
            overall_new_scores.append(new_scores)
        
        # Calculate and store overall statistics
        total_overlap_pct_original = 100 * total_overlap / total_original if total_original > 0 else 0
        total_overlap_pct_new = 100 * total_overlap / total_new if total_new > 0 else 0
        
        overall_orig = torch.cat(overall_original_scores)
        overall_new = torch.cat(overall_new_scores)
        
        overall_orig_stats = {
            "mean": overall_orig.mean().item(),
            "std": overall_orig.std().item(),
            "min": overall_orig.min().item(),
            "max": overall_orig.max().item()
        }
        
        overall_new_stats = {
            "mean": overall_new.mean().item(),
            "std": overall_new.std().item(),
            "min": overall_new.min().item(),
            "max": overall_new.max().item()
        }
        
        comparison_results["overall"] = {
            "original_count": total_original,
            "new_count": total_new,
            "overlap": total_overlap,
            "overlap_pct_original": total_overlap_pct_original,
            "overlap_pct_new": total_overlap_pct_new,
            "original_score_stats": overall_orig_stats,
            "new_score_stats": overall_new_stats
        }
        
        # Print overall summary
        print("\nOverall Summary:")
        print(f"Total neurons - Original: {total_original}, New: {total_new}, Overlap: {total_overlap}")
        print(f"Overall overlap % - Of original preserved: {total_overlap_pct_original:.2f}%, Of new from original: {total_overlap_pct_new:.2f}%")
        print(f"Overall original scores - Mean: {overall_orig_stats['mean']:.4f}, Std: {overall_orig_stats['std']:.4f}, Min: {overall_orig_stats['min']:.4f}, Max: {overall_orig_stats['max']:.4f}")
        print(f"Overall new scores     - Mean: {overall_new_stats['mean']:.4f}, Std: {overall_new_stats['std']:.4f}, Min: {overall_new_stats['min']:.4f}, Max: {overall_new_stats['max']:.4f}")

        # Save results to JSON file
        results_file = os.path.join(args.results_path, "neuron_comparison_results.json")
        stats_file_new = os.path.join(args.results_path, "new_scores_stats.json") 
        stats_file_orig = os.path.join(args.results_path, "original_scores_stats.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump(comparison_results, f, indent=4)
        with open(stats_file_new, "w") as f:
            json.dump(overall_new_stats, f, indent=4)
        with open(stats_file_orig, "w") as f:
            json.dump(overall_orig_stats, f, indent=4)
            
        print(f"\nResults saved to {results_file}")
        print(f"New scores stats saved to {stats_file_new}")
        print(f"Original scores stats saved to {stats_file_orig}")

        # Clear memory after Step 3
        torch.cuda.empty_cache()
        print(f"GPU memory after Step 3: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "Step 3 complete")


if __name__ == "__main__":
    main()
