#!/bin/bash
################################################################################
# Quick test with minimal resources to verify pipeline works
# Expected runtime: ~30-60 minutes
################################################################################

set -e  # Exit on error

echo "======================================================================"
echo "QUICK TEST - Experiment 1 Only"
echo "======================================================================"
echo "Settings: 1% neurons, 32 samples, 1 epoch, no ASR"
echo "Expected: 30-60 minutes"
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Quick test configuration
export MODEL="llama2-7b-chat-hf"
export PRUNE_METHOD="wanda"
export SPARSITY_RATIO="0.01"
export TRAINING_DATA="alpaca_cleaned_no_safety"
export NUM_EPOCHS="1"
export LEARNING_RATE="2e-5"
export BATCH_SIZE="2"
export MAX_LENGTH="256"
export NSAMPLES="32"
export SEED="42"
export DEVICE="cuda:0"
export EVAL_ATTACK="false"

# Run neuron identification + Experiment 1 only
./scripts/00_setup_neuron_identification.sh
./scripts/01_run_experiment_1_frozen.sh

echo ""
echo "✓ Quick test complete! Check ./results/${MODEL}/experiment_1_frozen/"
echo ""
echo "To run full experiments: ./scripts/run_full_pipeline.sh"
