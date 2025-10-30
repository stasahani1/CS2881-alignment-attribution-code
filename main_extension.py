"""
Extension for Safety-Critical Neuron Analysis and Fine-tuning

This module implements experiments to understand why safety alignment is brittle:
1. Experiment 1: Frozen-Regime Fine-Tuning (Wanda Score Dynamics)
2. Experiment 2: Unfrozen Fine-Tuning (Safety Neuron Drift)

Based on "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications"
"""

import os
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set HuggingFace cache to /tmp/ to avoid filling workspace storage
os.environ["HF_HOME"] = "/tmp/huggingface"

from main import modeltype2path, get_llm
from lib.extension_utils import (
    SafetyNeuronAnalyzer,
    FineTuner,
    WeightDriftAnalyzer,
    ScoreDynamicsAnalyzer,
    capture_model_weights,
    evaluate_model_safety
)


def main():
    """
    Main pipeline for safety-critical neuron analysis and fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(
        description="Safety-Critical Neuron Analysis and Fine-tuning Extension"
    )

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "identify_safety_neurons",
            "identify_utility_neurons",
            "fine_tune_frozen",
            "fine_tune_unfrozen",
            "eval_score_dynamics",
            "eval_weight_drift",
        ],
        help="Task to perform"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="llama2-7b-chat-hf",
        help="Model name to analyze"
    )
    parser.add_argument(
        "--original_model_path",
        type=str,
        default=None,
        help="Path to original pre-fine-tuned model (for evaluation tasks)"
    )
    parser.add_argument(
        "--fine_tuned_model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (for evaluation tasks)"
    )

    # Neuron identification arguments
    parser.add_argument(
        "--prune_method",
        type=str,
        default="wanda",
        choices=["wandg", "wanda"],
        help="Method for identifying critical neurons"
    )
    parser.add_argument(
        "--prune_data",
        type=str,
        default="align_short",
        help="Dataset for neuron analysis (align/align_short for safety, alpaca_cleaned_no_safety for utility)"
    )
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.05,
        help="Fraction of neurons to identify as critical"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of samples for scoring"
    )

    # Fine-tuning arguments
    parser.add_argument(
        "--safety_masks",
        type=str,
        default=None,
        help="Path to safety neuron masks (.pt file)"
    )
    parser.add_argument(
        "--utility_masks",
        type=str,
        default=None,
        help="Path to utility neuron masks (.pt file)"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default="alpaca_cleaned_no_safety",
        help="Dataset for fine-tuning"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # Evaluation arguments
    parser.add_argument(
        "--original_safety_masks_path",
        type=str,
        default=None,
        help="Path to original safety neuron masks"
    )
    parser.add_argument(
        "--original_safety_scores_path",
        type=str,
        default=None,
        help="Path to original safety neuron scores"
    )
    parser.add_argument(
        "--fine_tuned_safety_scores_path",
        type=str,
        default=None,
        help="Path to fine-tuned safety neuron scores"
    )
    parser.add_argument(
        "--safety_masks_path",
        type=str,
        default=None,
        help="Path to safety masks for drift analysis"
    )
    parser.add_argument(
        "--utility_masks_path",
        type=str,
        default=None,
        help="Path to utility masks for drift analysis"
    )
    parser.add_argument(
        "--original_weights_path",
        type=str,
        default=None,
        help="Path to pre-fine-tuning weights snapshot"
    )
    parser.add_argument(
        "--eval_attack",
        action="store_true",
        help="Run ASR evaluation using eval_attack() from base codebase"
    )

    # Output arguments
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="/tmp/fine_tuned_model",
        help="Path to save fine-tuned model"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results",
        help="Path to save analysis results"
    )

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

    # ========================================================================
    # Task: Identify Safety-Critical Neurons
    # ========================================================================
    if args.task == "identify_safety_neurons":
        print("\n" + "=" * 70)
        print("TASK: Identify Safety-Critical Neurons")
        print("=" * 70)

        # Load model and tokenizer
        print(f"\nLoading model {args.model}...")
        model = get_llm(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            modeltype2path[args.model], use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Identify safety-critical neurons
        analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
        safety_masks, safety_scores = analyzer.identify_safety_critical_neurons(
            prune_method=args.prune_method,
            prune_data=args.prune_data,
            sparsity_ratio=args.sparsity_ratio,
            nsamples=args.nsamples,
            seed=args.seed,
        )

        # Save results
        save_dir = os.path.join(args.results_path, "safety_neurons")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(safety_masks, os.path.join(save_dir, "original_safety_masks.pt"))
        torch.save(safety_scores, os.path.join(save_dir, "original_safety_scores.pt"))

        print(f"\nSaved safety neuron data to {save_dir}")

        # Clear memory
        torch.cuda.empty_cache()

    # ========================================================================
    # Task: Identify Utility-Critical Neurons
    # ========================================================================
    elif args.task == "identify_utility_neurons":
        print("\n" + "=" * 70)
        print("TASK: Identify Utility-Critical Neurons")
        print("=" * 70)

        # Load model and tokenizer
        print(f"\nLoading model {args.model}...")
        model = get_llm(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            modeltype2path[args.model], use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Identify utility-critical neurons using utility dataset
        analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
        utility_masks, utility_scores = analyzer.identify_safety_critical_neurons(
            prune_method=args.prune_method,
            prune_data=args.prune_data,  # Should be "alpaca_cleaned_no_safety"
            sparsity_ratio=args.sparsity_ratio,
            nsamples=args.nsamples,
            seed=args.seed,
        )

        # Save results
        save_dir = os.path.join(args.results_path, "utility_neurons")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(utility_masks, os.path.join(save_dir, "original_utility_masks.pt"))
        torch.save(utility_scores, os.path.join(save_dir, "original_utility_scores.pt"))

        print(f"\nSaved utility neuron data to {save_dir}")

        # Clear memory
        torch.cuda.empty_cache()

    # ========================================================================
    # Task: Fine-tune with Frozen Safety Neurons (Experiment 1)
    # ========================================================================
    elif args.task == "fine_tune_frozen":
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Frozen-Regime Fine-Tuning")
        print("=" * 70)

        # Load model and tokenizer
        print(f"\nLoading model {args.model}...")
        model = get_llm(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            modeltype2path[args.model], use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load safety masks
        if args.safety_masks is None:
            raise ValueError("--safety_masks is required for fine_tune_frozen task")

        print(f"\nLoading safety masks from {args.safety_masks}")
        safety_masks = torch.load(args.safety_masks)

        # Fine-tune the model with frozen safety neurons
        fine_tuner = FineTuner(model, tokenizer, device, safety_masks)

        print("\nFine-tuning model with frozen safety-critical neurons...")
        fine_tuned_model = fine_tuner.fine_tune_model(
            training_data=args.training_data,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_save_path=args.model_save_path,
        )

        # Clear memory after fine-tuning
        torch.cuda.empty_cache()

        # Re-calculate safety-critical neuron scores on fine-tuned model
        print("\nRe-calculating Wanda scores on fine-tuned model...")
        analyzer = SafetyNeuronAnalyzer(fine_tuned_model, tokenizer, device)

        new_safety_masks, new_safety_scores = analyzer.identify_safety_critical_neurons(
            prune_method=args.prune_method,
            prune_data=args.prune_data,
            sparsity_ratio=args.sparsity_ratio,
            nsamples=args.nsamples,
            seed=args.seed,
        )

        # Save results
        save_dir = os.path.join(args.results_path, "safety_neurons")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            new_safety_masks, os.path.join(save_dir, "fine_tuned_safety_masks.pt")
        )
        torch.save(
            new_safety_scores, os.path.join(save_dir, "fine_tuned_safety_scores.pt")
        )

        print(f"\nSaved fine-tuned safety neuron data to {save_dir}")

        # Optionally evaluate ASR
        if args.eval_attack:
            asr_results = evaluate_model_safety(
                model_path=args.model_save_path,
                tokenizer=tokenizer,
                save_dir=os.path.join(args.results_path, "attack_results"),
                prefix="fine_tuned_frozen",
            )

            # Save ASR results
            with open(
                os.path.join(args.results_path, "asr_results_frozen.json"), "w"
            ) as f:
                json.dump(asr_results, f, indent=4)

        # Clear memory
        torch.cuda.empty_cache()

    # ========================================================================
    # Task: Fine-tune without Freezing (Experiment 2)
    # ========================================================================
    elif args.task == "fine_tune_unfrozen":
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Unfrozen Fine-Tuning (Weight Drift Tracking)")
        print("=" * 70)

        # Load model and tokenizer
        print(f"\nLoading model {args.model}...")
        model = get_llm(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            modeltype2path[args.model], use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Capture original weights before fine-tuning
        print("\nCapturing original model weights...")
        original_weights = capture_model_weights(model)

        # Save original weights
        weights_save_path = os.path.join(args.results_path, "original_weights.pt")
        os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
        torch.save(original_weights, weights_save_path)
        print(f"Saved original weights to {weights_save_path}")

        # Fine-tune the model WITHOUT freezing (pass empty masks)
        fine_tuner = FineTuner(model, tokenizer, device, safety_masks=None)

        print("\nFine-tuning model without freezing any neurons...")
        fine_tuned_model = fine_tuner.fine_tune_model(
            training_data=args.training_data,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_save_path=args.model_save_path,
        )

        print(f"\nFine-tuned model saved to {args.model_save_path}")

        # Optionally evaluate ASR
        if args.eval_attack:
            asr_results = evaluate_model_safety(
                model_path=args.model_save_path,
                tokenizer=tokenizer,
                save_dir=os.path.join(args.results_path, "attack_results"),
                prefix="fine_tuned_unfrozen",
            )

            # Save ASR results
            with open(
                os.path.join(args.results_path, "asr_results_unfrozen.json"), "w"
            ) as f:
                json.dump(asr_results, f, indent=4)

        # Clear memory
        torch.cuda.empty_cache()

    # ========================================================================
    # Task: Evaluate Score Dynamics (Experiment 1 Analysis)
    # ========================================================================
    elif args.task == "eval_score_dynamics":
        print("\n" + "=" * 70)
        print("EVALUATION: Score Dynamics Analysis (Experiment 1)")
        print("=" * 70)

        # Load score data
        print("\nLoading score data...")
        original_scores = torch.load(args.original_safety_scores_path)
        fine_tuned_scores = torch.load(args.fine_tuned_safety_scores_path)
        safety_masks = torch.load(args.original_safety_masks_path)

        # Analyze score distributions
        print("\nComparing score distributions...")
        score_dist = ScoreDynamicsAnalyzer.compare_score_distributions(
            original_scores, fine_tuned_scores, safety_masks
        )

        # Compute score drop percentages
        print("\nComputing score drop percentages...")
        score_drops = ScoreDynamicsAnalyzer.compute_score_drop_percentage(
            original_scores, fine_tuned_scores, safety_masks
        )

        # Compute statistics
        results = {
            "score_distributions": {},
            "score_drops_per_layer": score_drops,
        }

        for regime in ["original", "new"]:
            for category in ["all", "safety"]:
                tensor = score_dist[regime][category]
                if len(tensor) > 0:
                    results["score_distributions"][f"{regime}_{category}"] = {
                        "mean": tensor.mean().item(),
                        "std": tensor.std().item(),
                        "median": tensor.median().item(),
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                        "count": len(tensor),
                    }

        # Print summary
        print("\n" + "=" * 70)
        print("SCORE DYNAMICS SUMMARY")
        print("=" * 70)

        print("\nOriginal Safety Neurons:")
        if "original_safety" in results["score_distributions"]:
            stats = results["score_distributions"]["original_safety"]
            print(
                f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, "
                f"Median: {stats['median']:.6f}"
            )

        print("\nFine-tuned Safety Neurons:")
        if "new_safety" in results["score_distributions"]:
            stats = results["score_distributions"]["new_safety"]
            print(
                f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, "
                f"Median: {stats['median']:.6f}"
            )

        # Compute overall score drop
        if (
            "original_safety" in results["score_distributions"]
            and "new_safety" in results["score_distributions"]
        ):
            orig_mean = results["score_distributions"]["original_safety"]["mean"]
            new_mean = results["score_distributions"]["new_safety"]["mean"]
            pct_change = ((new_mean - orig_mean) / orig_mean) * 100
            print(f"\nOverall Score Change: {pct_change:+.2f}%")

            if pct_change < -10:
                print(
                    "  → Supports Hypothesis A: Representational drift causing score drops"
                )
            else:
                print(
                    "  → Supports Hypothesis B: Scores remain stable despite freezing"
                )

        # Save results
        results_file = os.path.join(args.results_path, "score_dynamics_analysis.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to {results_file}")

    # ========================================================================
    # Task: Evaluate Weight Drift (Experiment 2 Analysis)
    # ========================================================================
    elif args.task == "eval_weight_drift":
        print("\n" + "=" * 70)
        print("EVALUATION: Weight Drift Analysis (Experiment 2)")
        print("=" * 70)

        # Load weight data
        print("\nLoading weight data...")
        original_weights = torch.load(args.original_weights_path)

        print(f"Loading fine-tuned model from {args.fine_tuned_model_path}...")
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(
            args.fine_tuned_model_path
        )
        fine_tuned_weights = capture_model_weights(fine_tuned_model)

        # Load masks
        safety_masks = torch.load(args.safety_masks_path)
        utility_masks = (
            torch.load(args.utility_masks_path) if args.utility_masks_path else None
        )

        # Analyze drift
        print("\nAnalyzing weight drift by neuron category...")
        drift_analyzer = WeightDriftAnalyzer(original_weights, fine_tuned_weights)

        drift_results = drift_analyzer.analyze_drift_by_category(
            safety_masks=safety_masks,
            utility_masks=utility_masks,
            random_sample_ratio=0.05,
        )

        # Compute statistics
        results = {}
        for category in ["safety", "utility", "random"]:
            results[category] = {
                "cosine_similarity": WeightDriftAnalyzer.compute_statistics(
                    drift_results[category]["cosine_sim"]
                ),
                "l2_distance": WeightDriftAnalyzer.compute_statistics(
                    drift_results[category]["l2_dist"]
                ),
            }

        # Print summary
        print("\n" + "=" * 70)
        print("WEIGHT DRIFT SUMMARY")
        print("=" * 70)

        for category in ["safety", "utility", "random"]:
            if results[category]["cosine_similarity"]["count"] > 0:
                cos_stats = results[category]["cosine_similarity"]
                l2_stats = results[category]["l2_distance"]

                print(f"\n{category.upper()} Neurons:")
                print(
                    f"  Cosine Similarity: Mean={cos_stats['mean']:.4f}, "
                    f"Std={cos_stats['std']:.4f}, Median={cos_stats['median']:.4f}"
                )
                print(
                    f"  L2 Distance: Mean={l2_stats['mean']:.4f}, "
                    f"Std={l2_stats['std']:.4f}, Median={l2_stats['median']:.4f}"
                )
                print(f"  Count: {cos_stats['count']}")

        # Hypothesis testing
        print("\n" + "=" * 70)
        print("HYPOTHESIS TESTING")
        print("=" * 70)

        if (
            results["safety"]["cosine_similarity"]["count"] > 0
            and results["random"]["cosine_similarity"]["count"] > 0
        ):
            safety_cos = results["safety"]["cosine_similarity"]["mean"]
            random_cos = results["random"]["cosine_similarity"]["mean"]

            print(
                f"\nSafety neurons cosine similarity: {safety_cos:.4f}"
            )
            print(f"Random neurons cosine similarity: {random_cos:.4f}")

            if safety_cos < random_cos - 0.05:  # Safety drifts more
                print(
                    "\n→ Supports Hypothesis C: Safety-critical neurons are fragile (high drift)"
                )
            else:
                print(
                    "\n→ Supports Hypothesis D: Safety neurons stable, "
                    "safety degradation from recontextualization"
                )

        # Save results
        results_file = os.path.join(args.results_path, "weight_drift_analysis.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to {results_file}")

        # Clear memory
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("TASK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
