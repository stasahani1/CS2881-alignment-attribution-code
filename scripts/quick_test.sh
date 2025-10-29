#!/bin/bash
################################################################################
# Script: quick_test.sh
# Description: Quick test with minimal resources to verify pipeline works
#              Use this before running full experiments
################################################################################

set -e  # Exit on error

echo "======================================================================"
echo "QUICK TEST MODE"
echo "======================================================================"
echo ""
echo "This will run the pipeline with minimal settings to test functionality."
echo "Settings:"
echo "  - Sparsity: 0.01 (1% of neurons)"
echo "  - Samples: 32 (minimal)"
echo "  - Epochs: 1"
echo "  - Batch size: 2"
echo "  - Max length: 256"
echo "  - No ASR evaluation"
echo ""
echo "Expected runtime: ~30-60 minutes (depends on GPU)"
echo ""
read -p "Press Enter to start quick test, or Ctrl+C to cancel..."
echo ""

# ============================================================================
# Quick Test Configuration
# ============================================================================
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

# ============================================================================
# Run Pipeline
# ============================================================================
./scripts/run_full_pipeline.sh

echo ""
echo "======================================================================"
echo "QUICK TEST COMPLETE"
echo "======================================================================"
echo ""
echo "If this completed successfully, you can now run full experiments with:"
echo "  ./scripts/run_full_pipeline.sh"
echo ""
echo "Or configure settings in config_example.sh and run:"
echo "  source scripts/config_example.sh"
echo "  ./scripts/run_full_pipeline.sh"
echo "======================================================================"
