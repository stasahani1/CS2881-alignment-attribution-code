#!/usr/bin/env python3
"""
Quick test script to verify the optimized neuron group functions work correctly.
Tests with small synthetic data to ensure logic is correct.
"""

import numpy as np
import torch
from identify_neuron_groups import get_topk_neurons, get_set_difference_neurons, get_random_neurons


def test_get_topk_neurons():
    """Test top-k neuron selection"""
    print("\n" + "="*60)
    print("Testing get_topk_neurons()")
    print("="*60)

    # Create small test data: 2 layers with known scores
    scores = {
        'layer_0_q': torch.tensor([[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]]),  # 6 neurons, max=6.0
        'layer_1_k': torch.tensor([[7.0, 8.0],
                                    [9.0, 10.0]]),       # 4 neurons, max=10.0
    }
    # Total: 10 neurons, top 20% = 2 neurons
    # Expected: (layer_1_k, 1, 1) with score=10.0 and (layer_1_k, 1, 0) with score=9.0

    result = get_topk_neurons(scores, k=0.2)

    print(f"\nResult: {result}")
    assert len(result) == 2, f"Expected 2 neurons, got {len(result)}"

    # Check that the top neurons are from layer_1_k (which has highest scores)
    layer_names = [name for name, _, _ in result]
    assert all(name == 'layer_1_k' for name in layer_names), "Top neurons should be from layer_1_k"

    print("✓ Test passed: get_topk_neurons() working correctly")


def test_get_set_difference_neurons():
    """Test set difference neuron selection"""
    print("\n" + "="*60)
    print("Testing get_set_difference_neurons()")
    print("="*60)

    # Create test data where some neurons are high in safety but not utility
    safety_scores = {
        'layer_0_q': torch.tensor([[1.0, 2.0, 9.0],   # High score at (0,2)
                                    [3.0, 8.0, 4.0]]),  # High score at (1,1)
    }

    utility_scores = {
        'layer_0_q': torch.tensor([[10.0, 1.0, 2.0],  # High score at (0,0)
                                    [1.0, 3.0, 1.0]]),
    }

    # With p=0.2 (1.2/6 ≈ 1) and q=0.4 (2.4/6 ≈ 2):
    # Top ~1/6 utility: (0,0) with score=10.0  [int(6*0.2) = 1]
    # Top ~2/6 safety: (0,2) with score=9.0 and (1,1) with score=8.0  [int(6*0.4) = 2]
    # Set diff should be: (0,2) and (1,1) since neither is (0,0)

    result = get_set_difference_neurons(safety_scores, utility_scores, p=0.2, q=0.4)

    print(f"\nResult: {result}")
    assert len(result) == 2, f"Expected 2 neurons, got {len(result)}"

    # Convert to set for easy checking
    result_set = {(row, col) for _, row, col in result}
    expected_positions = {(0, 2), (1, 1)}

    assert result_set == expected_positions, f"Expected {expected_positions}, got {result_set}"

    print("✓ Test passed: get_set_difference_neurons() working correctly")


def test_get_random_neurons():
    """Test random neuron selection"""
    print("\n" + "="*60)
    print("Testing get_random_neurons()")
    print("="*60)

    scores = {
        'layer_0_q': torch.tensor([[1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0]]),  # 6 neurons
        'layer_1_k': torch.tensor([[7.0, 8.0],
                                    [9.0, 10.0]]),       # 4 neurons
    }
    # Total: 10 neurons

    result = get_random_neurons(scores, num_neurons=5, seed=42)

    print(f"\nResult: {result}")
    assert len(result) == 5, f"Expected 5 neurons, got {len(result)}"

    # Check all neurons are unique
    assert len(set(result)) == 5, "Random neurons should be unique"

    # Check all neurons are valid (within bounds)
    for layer_name, row, col in result:
        assert layer_name in scores, f"Invalid layer: {layer_name}"
        shape = scores[layer_name].shape
        assert 0 <= row < shape[0], f"Row {row} out of bounds for {layer_name}"
        assert 0 <= col < shape[1], f"Col {col} out of bounds for {layer_name}"

    # Test reproducibility with same seed
    result2 = get_random_neurons(scores, num_neurons=5, seed=42)
    assert result == result2, "Same seed should produce same results"

    # Test different seed produces different results
    result3 = get_random_neurons(scores, num_neurons=5, seed=99)
    assert result != result3, "Different seed should produce different results"

    print("✓ Test passed: get_random_neurons() working correctly")


def test_performance_estimate():
    """Estimate performance on realistic data size"""
    print("\n" + "="*60)
    print("Performance Estimation")
    print("="*60)

    # Simulate realistic sizes for LLaMA-2-7B
    # Typical layer sizes: q_proj (4096x4096), mlp.up_proj (4096x11008), etc.
    import time

    # Create one realistic-sized layer
    large_layer = torch.randn(4096, 11008)  # ~45M parameters

    scores = {
        'layer_0_up_proj': large_layer,
    }

    print(f"Testing with 1 layer of size {large_layer.shape} ({large_layer.numel():,} neurons)")

    # Test top-k
    start = time.time()
    result = get_topk_neurons(scores, k=0.01)
    elapsed = time.time() - start
    print(f"\ntop-k (1%): {len(result):,} neurons in {elapsed:.3f}s")

    # Extrapolate to 224 layers
    estimated_total = elapsed * 224
    print(f"Estimated time for 224 layers: {estimated_total:.1f}s = {estimated_total/60:.1f} minutes")

    # Test random
    start = time.time()
    result = get_random_neurons(scores, num_neurons=450000, seed=0)
    elapsed = time.time() - start
    print(f"\nRandom (450K neurons): {len(result):,} neurons in {elapsed:.3f}s")
    estimated_total = elapsed * 224
    print(f"Estimated time for 224 layers: {estimated_total:.1f}s = {estimated_total/60:.1f} minutes")

    print("\n✓ Performance test complete")


if __name__ == "__main__":
    print("="*60)
    print("Testing Optimized Neuron Group Functions")
    print("="*60)

    test_get_topk_neurons()
    test_get_set_difference_neurons()
    test_get_random_neurons()
    test_performance_estimate()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
