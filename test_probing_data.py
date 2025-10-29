"""
Quick test script to verify probing data loading works correctly.
"""

import sys
sys.path.append('.')

from probing import load_harmful_instructions, load_harmless_instructions

print("Testing data loading for probing...")
print("="*80)

# Test harmful instructions
print("\n1. Loading harmful instructions...")
harmful = load_harmful_instructions('data/advbench.txt', n_samples=5)
print(f"   Loaded {len(harmful)} harmful instructions")
print("\n   Sample harmful instructions:")
for i, inst in enumerate(harmful[:3], 1):
    print(f"   {i}. {inst[:80]}...")

# Test harmless instructions
print("\n2. Loading harmless instructions...")
harmless = load_harmless_instructions('data/alpaca_cleaned_no_safety_train.csv', n_samples=5)
print(f"   Loaded {len(harmless)} harmless instructions")
print("\n   Sample harmless instructions:")
for i, inst in enumerate(harmless[:3], 1):
    print(f"   {i}. {inst[:80]}...")

print("\n" + "="*80)
print("Data loading test completed successfully!")
print("="*80)
