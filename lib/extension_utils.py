import torch
import torch.nn as nn
from typing import Dict
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from vllm import LLM

from .data import get_loaders
from .prune import check_sparsity, find_layers
from .model_wrapper import make_Act, revert_Act_to_Linear, ActLinear, no_act_recording
from .eval import eval_attack

class SafetyNeuronAnalyzer:
    """
    Analyzes and manages safety-critical neurons in language models.
    """
    
    def __init__(self, model, tokenizer, device=torch.device("cuda:0")):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.safety_critical_neurons = {}
        self.safety_critical_scores = {}
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
            Tuple of (masks, scores) where:
            - masks: Dictionary mapping layer names to boolean masks indicating safety-critical neurons
            - scores: Dictionary mapping layer names to importance scores for all neurons
        """
        print(f"Calculating {prune_method} scores on {prune_data} dataset...")

        # Clear GPU cache before starting
        torch.cuda.empty_cache()

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
        safety_scores = {}
        
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

                # Clear batch tensors immediately to free memory
                del inp, tar, loss

            # Extract scores and create masks for this layer
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    if prune_method == "wandg":
                        # SNIP: Use gradient magnitude as importance score
                        scores_2d = torch.abs(module.base.weight.grad).cpu()
                    elif prune_method == "wanda":
                        # Wanda: Use |weight| * sqrt(activation_norm)
                        scores_2d = (torch.abs(module.base.weight.data) * torch.sqrt(
                            module.activation_norms.reshape((1, -1))
                        )).cpu()
                    else:
                        raise ValueError(f"Unsupported prune method: {prune_method}")

                    # Flatten scores for selection
                    flat_scores = scores_2d.flatten()

                    # Select top-k% as safety-critical neurons
                    num_to_select = int(flat_scores.numel() * sparsity_ratio)

                    if num_to_select > 0:
                        # Get top-k indices and their scores
                        # This is much more memory efficient than storing all scores
                        topk_scores, topk_indices = torch.topk(flat_scores, num_to_select, largest=True)

                        # Store only safety-critical indices and their scores
                        # Storage: ~838K entries (5%) vs 16.7M entries (100%) = 95% reduction
                        safety_scores[name] = {
                            'indices': topk_indices,  # Which neurons are safety-critical
                            'scores': topk_scores,     # Their importance scores
                            'shape': scores_2d.shape,  # Original shape for reconstruction
                        }

                        # Create mask for freezing (still need 2D for gradient hooks)
                        mask = torch.zeros_like(scores_2d, dtype=torch.bool)
                        mask.view(-1)[topk_indices] = True
                    else:
                        # If sparsity_ratio is 0, no neurons are safety-critical
                        safety_scores[name] = {
                            'indices': torch.tensor([], dtype=torch.long),
                            'scores': torch.tensor([]),
                            'shape': scores_2d.shape,
                        }
                        mask = torch.zeros_like(scores_2d, dtype=torch.bool)

                    safety_masks[name] = mask

            # Clear gradients and cache after each layer
            model.zero_grad()
            torch.cuda.empty_cache()
        
        # Convert back to regular model (non-destructive)
        model = revert_Act_to_Linear(model)
        
        print(f"Identified safety-critical neurons in {len(safety_masks)} layers")
        total_critical = sum(mask.sum().item() for mask in safety_masks.values())
        total_neurons = sum(mask.numel() for mask in safety_masks.values())
        print(f"Total safety-critical neurons: {total_critical}/{total_neurons} ({100*total_critical/total_neurons:.2f}%)")
        
        print("Checking model sparsity...")
        sparsity = check_sparsity(self.model)
        print(f"Overall model sparsity: {sparsity:.4f}")
        
        self.safety_critical_neurons = safety_masks
        self.safety_critical_scores = safety_scores
        return safety_masks, safety_scores



class FineTuner:
    """
    Handles fine-tuning of models with frozen safety-critical neurons.
    """
    
    def __init__(self, model, tokenizer, device=torch.device("cuda:0"), safety_masks=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.safety_masks = safety_masks
    
    def _freeze_safety_critical_neurons(self) -> nn.Module:
        """
        Freeze safety-critical neurons by setting their gradients to zero.
        
        Args:
            model: The model to freeze neurons in
            
        Returns:
            The model with frozen safety-critical neurons
        """
        if self.safety_masks is None:
            print("Warning: No safety masks provided. Skipping neuron freezing.")
            return self.model
            
        print("Freezing safety-critical neurons...")
        
        frozen_count = 0
        total_count = 0
        
        # Use existing find_layers function for consistency
        linear_layers = find_layers(self.model)
        for name, module in linear_layers.items():
            if name in self.safety_masks:
                mask = self.safety_masks[name]
                
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
        return self.model
    
    def fine_tune_model(self, 
                       training_data: str = "alpaca_cleaned_no_safety",
                       num_epochs: int = 3,
                       learning_rate: float = 2e-5,
                       batch_size: int = 4,
                       max_length: int = 2048,
                       model_save_path: str = "./models/fine_tuned_model") -> nn.Module:
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
        # Build dataset iteratively to handle large tensors properly
        dataset_list = []
        for inp, tar in trainloader:
            # Convert tensors to lists and handle truncation
            inp_ids = inp.squeeze().tolist()
            tar_ids = tar.squeeze().tolist()

            # Truncate if necessary
            if len(inp_ids) > max_length:
                inp_ids = inp_ids[:max_length]
                tar_ids = tar_ids[:max_length]

            dataset_list.append({
                "input_ids": inp_ids,
                "labels": tar_ids
            })

        dataset = Dataset.from_list(dataset_list)
        
        # Training arguments
        # Use gradient accumulation for memory efficiency on smaller GPUs
        # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
        training_args = TrainingArguments(
            output_dir=model_save_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,  # Use mixed precision for memory efficiency
            gradient_checkpointing=True,  # Set False if GPU > 60 GB
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Freeze safety-critical neurons before training
        self.model = self._freeze_safety_critical_neurons()

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

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
        trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

        print(f"Fine-tuning completed. Model saved to {model_save_path}")
        return self.model


class WeightDriftAnalyzer:
    """
    Analyzes weight drift for different neuron categories during fine-tuning.
    Used for Experiment 2 (Unfrozen Fine-Tuning).
    """

    def __init__(self, original_weights: Dict[str, torch.Tensor],
                 fine_tuned_weights: Dict[str, torch.Tensor]):
        """
        Args:
            original_weights: Dictionary mapping layer names to pre-fine-tuning weights
            fine_tuned_weights: Dictionary mapping layer names to post-fine-tuning weights
        """
        self.original_weights = original_weights
        self.fine_tuned_weights = fine_tuned_weights

    def compute_cosine_similarity(self, layer_name: str, neuron_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute cosine similarity between original and fine-tuned weights.

        Args:
            layer_name: Name of the layer
            neuron_mask: Boolean mask indicating which neurons to analyze (None = all)

        Returns:
            Tensor of cosine similarities per neuron
        """
        if layer_name not in self.original_weights or layer_name not in self.fine_tuned_weights:
            raise ValueError(f"Layer {layer_name} not found in weight dictionaries")

        orig = self.original_weights[layer_name]
        fine = self.fine_tuned_weights[layer_name]

        # Apply mask if provided
        if neuron_mask is not None:
            orig = orig[neuron_mask]
            fine = fine[neuron_mask]

        # Compute cosine similarity per neuron (row-wise for weight matrices)
        # Shape: (out_features, in_features) -> compute similarity per output neuron
        orig_norm = torch.nn.functional.normalize(orig, p=2, dim=1)
        fine_norm = torch.nn.functional.normalize(fine, p=2, dim=1)

        # Cosine similarity: dot product of normalized vectors
        cosine_sim = (orig_norm * fine_norm).sum(dim=1)

        return cosine_sim

    def compute_l2_distance(self, layer_name: str, neuron_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute L2 distance between original and fine-tuned weights.

        Args:
            layer_name: Name of the layer
            neuron_mask: Boolean mask indicating which neurons to analyze (None = all)

        Returns:
            Tensor of L2 distances per neuron
        """
        if layer_name not in self.original_weights or layer_name not in self.fine_tuned_weights:
            raise ValueError(f"Layer {layer_name} not found in weight dictionaries")

        orig = self.original_weights[layer_name]
        fine = self.fine_tuned_weights[layer_name]

        # Apply mask if provided
        if neuron_mask is not None:
            orig = orig[neuron_mask]
            fine = fine[neuron_mask]

        # Compute L2 distance per neuron (row-wise)
        l2_dist = torch.norm(orig - fine, p=2, dim=1)

        return l2_dist

    def analyze_drift_by_category(self, safety_masks: Dict[str, torch.Tensor],
                                   utility_masks: Dict[str, torch.Tensor] = None,
                                   random_sample_ratio: float = 0.05) -> Dict:
        """
        Compare drift for safety-critical, utility-critical, and random neurons.

        Args:
            safety_masks: Dictionary mapping layer names to safety-critical neuron masks
            utility_masks: Dictionary mapping layer names to utility-critical neuron masks
            random_sample_ratio: Ratio of random neurons to sample for comparison

        Returns:
            Dictionary with drift statistics per category
        """
        results = {
            "safety": {"cosine_sim": [], "l2_dist": []},
            "utility": {"cosine_sim": [], "l2_dist": []},
            "random": {"cosine_sim": [], "l2_dist": []}
        }

        for layer_name in safety_masks.keys():
            if layer_name not in self.original_weights:
                continue

            # Safety neurons
            safety_mask = safety_masks[layer_name]
            if safety_mask.sum() > 0:
                cos_sim = self.compute_cosine_similarity(layer_name, safety_mask)
                l2_dist = self.compute_l2_distance(layer_name, safety_mask)
                results["safety"]["cosine_sim"].append(cos_sim)
                results["safety"]["l2_dist"].append(l2_dist)

            # Utility neurons
            if utility_masks is not None and layer_name in utility_masks:
                utility_mask = utility_masks[layer_name]
                if utility_mask.sum() > 0:
                    cos_sim = self.compute_cosine_similarity(layer_name, utility_mask)
                    l2_dist = self.compute_l2_distance(layer_name, utility_mask)
                    results["utility"]["cosine_sim"].append(cos_sim)
                    results["utility"]["l2_dist"].append(l2_dist)

            # Random neurons (excluding safety and utility)
            combined_mask = safety_mask.clone()
            if utility_masks is not None and layer_name in utility_masks:
                combined_mask = torch.logical_or(combined_mask, utility_masks[layer_name])

            random_mask = torch.logical_not(combined_mask)
            num_random = int(random_mask.sum().item() * random_sample_ratio)

            if num_random > 0:
                # Sample random neurons
                random_indices = torch.where(random_mask)[0]
                sampled_indices = random_indices[torch.randperm(len(random_indices))[:num_random]]
                random_sample_mask = torch.zeros_like(random_mask)
                random_sample_mask[sampled_indices] = True

                cos_sim = self.compute_cosine_similarity(layer_name, random_sample_mask)
                l2_dist = self.compute_l2_distance(layer_name, random_sample_mask)
                results["random"]["cosine_sim"].append(cos_sim)
                results["random"]["l2_dist"].append(l2_dist)

        # Concatenate results across all layers
        for category in results.keys():
            if len(results[category]["cosine_sim"]) > 0:
                results[category]["cosine_sim"] = torch.cat(results[category]["cosine_sim"])
                results[category]["l2_dist"] = torch.cat(results[category]["l2_dist"])
            else:
                results[category]["cosine_sim"] = torch.tensor([])
                results[category]["l2_dist"] = torch.tensor([])

        return results

    @staticmethod
    def compute_statistics(tensor: torch.Tensor) -> Dict:
        """
        Compute summary statistics for a tensor.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary with mean, std, median, min, max
        """
        if len(tensor) == 0:
            return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0, "count": 0}

        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "median": tensor.median().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "count": len(tensor)
        }


class ScoreDynamicsAnalyzer:
    """
    Analyzes how Wanda scores change for frozen safety-critical neurons.
    Used for Experiment 1 (Frozen-Regime Fine-Tuning).
    """

    @staticmethod
    def compare_score_distributions(original_scores: Dict[str, torch.Tensor],
                                     new_scores: Dict[str, torch.Tensor],
                                     safety_masks: Dict[str, torch.Tensor]) -> Dict:
        """
        Compare score distributions before and after fine-tuning for safety neurons.

        Args:
            original_scores: Pre-fine-tuning Wanda scores per layer
            new_scores: Post-fine-tuning Wanda scores per layer
            safety_masks: Boolean masks indicating safety-critical neurons

        Returns:
            Dictionary with statistical comparison
        """
        results = {
            "original": {"all": [], "safety": []},
            "new": {"all": [], "safety": []}
        }

        for layer_name in safety_masks.keys():
            if layer_name not in original_scores or layer_name not in new_scores:
                continue

            # Handle sparse storage format (dict with indices/scores)
            orig_data = original_scores[layer_name]
            new_data = new_scores[layer_name]
            mask = safety_masks[layer_name].flatten()

            # Extract scores from sparse format
            if isinstance(orig_data, dict):
                orig_indices = orig_data['indices']
                orig_scores = orig_data['scores']
            else:
                # Fallback for old format (full scores)
                orig_scores = orig_data.flatten()
                orig_indices = torch.arange(len(orig_scores))

            if isinstance(new_data, dict):
                new_indices = new_data['indices']
                new_scores = new_data['scores']
            else:
                # Fallback for old format (full scores)
                new_scores = new_data.flatten()
                new_indices = torch.arange(len(new_scores))

            # For "all neurons" comparison, we only have the top-k from each
            # This is sufficient since we care about safety-critical neurons
            results["original"]["all"].append(orig_scores)
            results["new"]["all"].append(new_scores)

            # Safety-critical neurons: those in original top-k
            # We compare their scores in original vs fine-tuned model
            if len(orig_indices) > 0:
                results["original"]["safety"].append(orig_scores)
                # For new scores, we need to find the same indices
                # Create a mapping from indices to scores in new model
                if isinstance(new_data, dict):
                    # Build full score array for indexing (only for originally safety-critical neurons)
                    new_full = torch.zeros(orig_data['shape'][0] * orig_data['shape'][1])
                    new_full[new_indices] = new_scores
                    # Get scores for originally safety-critical neurons
                    results["new"]["safety"].append(new_full[orig_indices])
                else:
                    results["new"]["safety"].append(new_scores[mask])

        # Concatenate across layers
        for regime in ["original", "new"]:
            for category in ["all", "safety"]:
                if len(results[regime][category]) > 0:
                    results[regime][category] = torch.cat(results[regime][category])
                else:
                    results[regime][category] = torch.tensor([])

        return results

    @staticmethod
    def compute_score_drop_percentage(original_scores: Dict[str, torch.Tensor],
                                       new_scores: Dict[str, torch.Tensor],
                                       safety_masks: Dict[str, torch.Tensor]) -> Dict:
        """
        Compute percentage drop in Wanda scores for frozen safety neurons.

        Args:
            original_scores: Pre-fine-tuning Wanda scores per layer
            new_scores: Post-fine-tuning Wanda scores per layer
            safety_masks: Boolean masks indicating safety-critical neurons

        Returns:
            Dictionary with drop percentages per layer and overall
        """
        results = {}

        for layer_name in safety_masks.keys():
            if layer_name not in original_scores or layer_name not in new_scores:
                continue

            # Handle sparse storage format
            orig_data = original_scores[layer_name]
            new_data = new_scores[layer_name]
            mask = safety_masks[layer_name].flatten()

            if mask.sum() == 0:
                continue

            # Extract scores from sparse format
            if isinstance(orig_data, dict):
                orig_indices = orig_data['indices']
                orig_safety = orig_data['scores']

                # Get new scores for the same indices
                if isinstance(new_data, dict):
                    new_full = torch.zeros(orig_data['shape'][0] * orig_data['shape'][1])
                    new_full[new_data['indices']] = new_data['scores']
                    new_safety = new_full[orig_indices]
                else:
                    new_safety = new_data.flatten()[orig_indices]
            else:
                # Fallback for old format
                orig = orig_data.flatten()
                new = new_data.flatten() if not isinstance(new_data, dict) else new_data.flatten()
                orig_safety = orig[mask]
                new_safety = new[mask]

            # Avoid division by zero
            orig_safety_safe = torch.where(orig_safety == 0, torch.ones_like(orig_safety) * 1e-10, orig_safety)

            # Percentage change: (new - orig) / orig * 100
            pct_change = ((new_safety - orig_safety) / orig_safety_safe * 100)

            results[layer_name] = {
                "mean_pct_change": pct_change.mean().item(),
                "median_pct_change": pct_change.median().item(),
                "std_pct_change": pct_change.std().item(),
                "original_mean_score": orig_safety.mean().item(),
                "new_mean_score": new_safety.mean().item()
            }

        return results


def capture_model_weights(model) -> Dict[str, torch.Tensor]:
    """
    Capture a snapshot of model weights for drift analysis.
    Reuses find_layers from lib.prune for consistency.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping layer names to weight tensors
    """
    weights = {}
    linear_layers = find_layers(model)

    for name, module in linear_layers.items():
        if hasattr(module, 'weight'):
            weights[name] = module.weight.data.clone().detach().cpu()

    return weights


def evaluate_model_safety(model_path: str, tokenizer, save_dir: str,
                          prefix: str = "") -> Dict:
    """
    Evaluate model safety using ASR (Attack Success Rate) on AdvBench dataset.
    Reuses eval_attack() from lib.eval for consistency with base codebase.

    Args:
        model_path: Path to model (for vLLM loading)
        tokenizer: Tokenizer for the model
        save_dir: Directory to save attack results
        prefix: Prefix for result filenames (e.g., "original", "fine_tuned")

    Returns:
        Dictionary with ASR scores for different attack configurations
    """
    import os

    print(f"\n{'='*50}")
    print(f"Evaluating model safety: {prefix}")
    print(f"{'='*50}")

    # Load model with vLLM for efficient generation
    # Note: eval_attack expects a vLLM model
    vllm_model = LLM(model=model_path, tensor_parallel_size=1)

    results = {}

    # Test configurations from base paper
    configs = [
        {"name": "inst_basic", "add_sys_prompt": True, "include_inst": True, "gcg": False},
        {"name": "inst_basic_no_sys", "add_sys_prompt": False, "include_inst": True, "gcg": False},
        {"name": "no_inst_basic", "add_sys_prompt": True, "include_inst": False, "gcg": False},
        {"name": "no_inst_basic_no_sys", "add_sys_prompt": False, "include_inst": False, "gcg": False},
        {"name": "gcg", "add_sys_prompt": False, "include_inst": True, "gcg": True},
    ]

    for config in configs:
        config_name = config["name"]
        print(f"\nRunning attack configuration: {config_name}")

        filename = os.path.join(save_dir, f"{prefix}_{config_name}.jsonl")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        asr_score = eval_attack(
            model=vllm_model,
            tokenizer=tokenizer,
            num_sampled=1,
            add_sys_prompt=config["add_sys_prompt"],
            prompt_template_style="base",
            do_sample=not config["gcg"],
            gcg=config["gcg"],
            include_inst=config["include_inst"],
            save_attack_res=True,
            filename=filename
        )

        results[config_name] = asr_score
        print(f"  ASR Score: {asr_score:.4f}")

    # Compute average ASR
    results["average_asr"] = sum(results.values()) / len(results)
    print(f"\nAverage ASR: {results['average_asr']:.4f}")

    return results
