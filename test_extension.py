#!/usr/bin/env python3
"""
Test script for the safety-critical neuron analysis extension.

This script performs basic tests to verify the implementation works correctly.
"""

import torch
import torch.nn as nn
import numpy as np
from extension import SafetyNeuronAnalyzer, FineTuner


class SimpleModel(nn.Module):
    """
    Simple test model for verification.
    """
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class SimpleTokenizer:
    """
    Simple tokenizer for testing.
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
    
    def __call__(self, text, return_tensors=None, **kwargs):
        # Simple tokenization for testing
        tokens = text.split()
        input_ids = torch.tensor([[hash(token) % 1000 for token in tokens]])
        return type('obj', (object,), {'input_ids': input_ids})()


def test_safety_neuron_analyzer():
    """
    Test the SafetyNeuronAnalyzer class.
    """
    print("Testing SafetyNeuronAnalyzer...")
    
    # Create test model
    model = SimpleModel()
    tokenizer = SimpleTokenizer()
    device = torch.device("cpu")
    
    analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
    
    # Test weight storage
    analyzer._store_original_weights(model)
    assert len(analyzer.original_weights) == 2  # Should have 2 linear layers
    print("✓ Weight storage test passed")
    
    # Test mask extraction (simulate pruned model)
    pruned_model = SimpleModel()
    # Simulate pruning by setting some weights to zero
    pruned_model.linear1.weight.data[0, :] = 0  # Prune first row
    pruned_model.linear2.weight.data[:, 0] = 0  # Prune first column
    
    masks = analyzer._extract_safety_masks(pruned_model)
    assert len(masks) == 2  # Should have masks for both layers
    assert masks['linear1'].shape == pruned_model.linear1.weight.shape
    assert masks['linear2'].shape == pruned_model.linear2.weight.shape
    print("✓ Mask extraction test passed")
    
    print("SafetyNeuronAnalyzer tests passed!\n")


def test_gradient_hooks():
    """
    Test the gradient hook mechanism for freezing neurons.
    """
    print("Testing gradient hooks...")
    
    # Create test model
    model = SimpleModel()
    tokenizer = SimpleTokenizer()
    device = torch.device("cpu")
    
    analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
    
    # Create fake safety masks
    analyzer.safety_critical_neurons = {
        'linear1': torch.zeros(model.linear1.weight.shape, dtype=torch.bool),
        'linear2': torch.zeros(model.linear2.weight.shape, dtype=torch.bool)
    }
    # Set some neurons as safety-critical
    analyzer.safety_critical_neurons['linear1'][0, :] = True  # First row
    analyzer.safety_critical_neurons['linear2'][:, 0] = True  # First column
    
    # Apply freezing
    frozen_model = analyzer.freeze_safety_critical_neurons(model)
    
    # Test that gradients are zeroed for frozen neurons
    x = torch.randn(1, 10, requires_grad=True)
    output = frozen_model(x)
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are zero for frozen neurons
    assert torch.allclose(frozen_model.linear1.weight.grad[0, :], torch.zeros_like(frozen_model.linear1.weight.grad[0, :]))
    assert torch.allclose(frozen_model.linear2.weight.grad[:, 0], torch.zeros_like(frozen_model.linear2.weight.grad[:, 0]))
    
    print("✓ Gradient hooks test passed")
    print("Gradient hooks tests passed!\n")


def test_fine_tuner():
    """
    Test the FineTuner class.
    """
    print("Testing FineTuner...")
    
    # Create test model
    model = SimpleModel()
    tokenizer = SimpleTokenizer()
    device = torch.device("cpu")
    
    fine_tuner = FineTuner(model, tokenizer, device)
    
    # Test that the fine_tuner is properly initialized
    assert fine_tuner.model == model
    assert fine_tuner.tokenizer == tokenizer
    assert fine_tuner.device == device
    
    print("✓ FineTuner initialization test passed")
    print("FineTuner tests passed!\n")


def test_integration():
    """
    Test the integration between components.
    """
    print("Testing integration...")
    
    # Create test model
    model = SimpleModel()
    tokenizer = SimpleTokenizer()
    device = torch.device("cpu")
    
    # Test analyzer initialization
    analyzer = SafetyNeuronAnalyzer(model, tokenizer, device)
    
    # Test fine_tuner initialization
    fine_tuner = FineTuner(model, tokenizer, device)
    
    # Test that both can work with the same model
    assert analyzer.model == fine_tuner.model
    assert analyzer.tokenizer == fine_tuner.tokenizer
    
    print("✓ Integration test passed")
    print("Integration tests passed!\n")


def run_all_tests():
    """
    Run all tests.
    """
    print("="*50)
    print("Running Safety-Critical Neuron Analysis Tests")
    print("="*50)
    
    try:
        test_safety_neuron_analyzer()
        test_gradient_hooks()
        test_fine_tuner()
        test_integration()
        
        print("="*50)
        print("All tests passed! ✓")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\nThe extension implementation is working correctly!")
        print("You can now use the extension.py module for your research.")
    else:
        print("\nSome tests failed. Please check the implementation.")

