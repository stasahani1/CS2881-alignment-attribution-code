import torch
import torch.nn as nn
from typing import Dict
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

from .data import get_loaders
from .prune import check_sparsity, find_layers
from .model_wrapper import make_Act, revert_Act_to_Linear, ActLinear, no_act_recording

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
                    
                    # Store all scores for this layer
                    safety_scores[name] = scores.clone()
                    
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
        def tokenize_function(examples):
            # This is a simplified tokenization 
            inputs = []
            targets = []
            for inp, tar in zip(examples["input_ids"], examples["labels"]):
                inputs.append(inp)
                targets.append(tar)
            return {"input_ids": inputs, "labels": targets}
        
        # Create dataset
        dataset_dict = {
            "input_ids": [inp.squeeze() for inp, _ in trainloader],
            "labels": [tar.squeeze() for _, tar in trainloader]
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
        
        # Freeze safety-critical neurons before training
        self.model = self._freeze_safety_critical_neurons()
        
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
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Fine-tuning completed. Model saved to {save_path}")
        return self.model
