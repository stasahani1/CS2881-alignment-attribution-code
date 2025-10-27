#!/usr/bin/env python3
"""
Example script demonstrating the usage of the safety-critical neuron analysis extension.

This script shows how to use the extension.py module to:
1. Identify safety-critical neurons
2. Freeze them during fine-tuning
3. Analyze changes in safety-critical patterns
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from extension import SafetyNeuronAnalyzer, FineTuner, recalculate_safety_scores


def run_example_analysis():
    """
    Run a complete example of the safety-critical neuron analysis pipeline.
    """
    print("="*60)
    print("Safety-Critical Neuron Analysis Example")
    print("="*60)
    
    # Configuration
    model_name = "llama2-7b-chat-hf"  # Adjust based on available models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.seqlen = model.config.max_position_embeddings
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model is available or adjust the model_name variable")
        return
    
    # Step 1: Identify safety-critical neurons
    print("\n" + "="*40)
    print("STEP 1: Identifying Safety-Critical Neurons")
    print("="*40)
    
    analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
    
    try:
        safety_masks = analyzer.identify_safety_critical_neurons(
            prune_method="wandg",  # Use SNIP scoring
            prune_data="align",     # Use safety dataset
            sparsity_ratio=0.1,     # Identify top 10% as safety-critical
            nsamples=64,           # Use 64 samples for analysis
            seed=42
        )
        
        print(f"Identified safety-critical neurons in {len(safety_masks)} layers")
        
    except Exception as e:
        print(f"Error in safety neuron identification: {e}")
        print("This might be due to missing datasets or model compatibility issues")
        return
    
    # Step 2: Freeze neurons and fine-tune
    print("\n" + "="*40)
    print("STEP 2: Freezing Neurons and Fine-tuning")
    print("="*40)
    
    try:
        # Freeze safety-critical neurons
        model = analyzer.freeze_safety_critical_neurons(model)
        
        # Fine-tune the model
        fine_tuner = FineTuner(model, tokenizer, device)
        fine_tuned_model = fine_tuner.fine_tune_model(
            training_data="alpaca_cleaned_no_safety",
            num_epochs=1,  # Use fewer epochs for example
            learning_rate=2e-5,
            batch_size=2,  # Smaller batch size for example
            max_length=256,
            save_path="./example_fine_tuned_model"
        )
        
        print("Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"Error in fine-tuning: {e}")
        print("This might be due to missing training data or memory constraints")
        return
    
    # Step 3: Recalculate safety scores
    print("\n" + "="*40)
    print("STEP 3: Recalculating Safety Scores")
    print("="*40)
    
    try:
        new_safety_masks = recalculate_safety_scores(
            fine_tuned_model,
            tokenizer,
            analyzer,
            prune_method="wandg",
            prune_data="align",
            sparsity_ratio=0.1
        )
        
        print("Safety score recalculation completed!")
        
    except Exception as e:
        print(f"Error in score recalculation: {e}")
        return
    
    # Step 4: Analysis and Results
    print("\n" + "="*40)
    print("STEP 4: Analysis Results")
    print("="*40)
    
    # Calculate statistics
    original_count = sum(mask.sum().item() for mask in safety_masks.values())
    new_count = sum(mask.sum().item() for mask in new_safety_masks.values())
    total_neurons = sum(mask.numel() for mask in safety_masks.values())
    
    print(f"Original safety-critical neurons: {original_count:,}")
    print(f"New safety-critical neurons: {new_count:,}")
    print(f"Total neurons analyzed: {total_neurons:,}")
    print(f"Change in safety-critical neurons: {new_count - original_count:,}")
    print(f"Percentage change: {100 * (new_count - original_count) / original_count:.2f}%")
    
    # Layer-wise analysis
    print(f"\nLayer-wise analysis:")
    print(f"{'Layer Name':<30} {'Original':<10} {'New':<10} {'Overlap':<10} {'Overlap %':<10}")
    print("-" * 70)
    
    for layer_name in safety_masks:
        if layer_name in new_safety_masks:
            original_mask = safety_masks[layer_name]
            new_mask = new_safety_masks[layer_name]
            
            original_count_layer = original_mask.sum().item()
            new_count_layer = new_mask.sum().item()
            overlap_count = (original_mask & new_mask).sum().item()
            overlap_percentage = 100 * overlap_count / max(original_count_layer, new_count_layer, 1)
            
            print(f"{layer_name:<30} {original_count_layer:<10} {new_count_layer:<10} {overlap_count:<10} {overlap_percentage:<10.1f}")
    
    # Save results
    print(f"\nSaving results...")
    os.makedirs("./example_results", exist_ok=True)
    
    torch.save(safety_masks, "./example_results/original_safety_masks.pt")
    torch.save(new_safety_masks, "./example_results/new_safety_masks.pt")
    
    results_summary = {
        "original_safety_critical_count": original_count,
        "new_safety_critical_count": new_count,
        "total_neurons": total_neurons,
        "change_count": new_count - original_count,
        "change_percentage": 100 * (new_count - original_count) / original_count,
        "model_name": model_name,
        "prune_method": "wandg",
        "prune_data": "align",
        "sparsity_ratio": 0.1
    }
    
    import json
    with open("./example_results/analysis_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print("Results saved to ./example_results/")
    print("\nAnalysis completed successfully!")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    try:
        import shutil
        if os.path.exists("./example_fine_tuned_model"):
            shutil.rmtree("./example_fine_tuned_model")
        print("Cleanup completed!")
    except Exception as e:
        print(f"Cleanup warning: {e}")


def run_minimal_example():
    """
    Run a minimal example that focuses on the core functionality.
    """
    print("="*60)
    print("Minimal Safety-Critical Neuron Analysis Example")
    print("="*60)
    
    # This is a simplified example that demonstrates the core concepts
    # without requiring full model loading and training
    
    print("This example demonstrates the core concepts:")
    print("1. Safety-critical neuron identification using SNIP/Wanda scores")
    print("2. Neuron freezing using gradient hooks")
    print("3. Fine-tuning with frozen neurons")
    print("4. Post-training analysis of safety-critical patterns")
    
    print("\nTo run the full example, ensure you have:")
    print("- A compatible model (e.g., llama2-7b-chat-hf)")
    print("- Required datasets (align, alpaca_cleaned_no_safety)")
    print("- Sufficient GPU memory")
    print("- All dependencies installed")
    
    print("\nFor the full example, run:")
    print("python example_usage.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        run_minimal_example()
    else:
        run_example_analysis()

