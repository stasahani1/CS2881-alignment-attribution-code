#!/bin/bash
################################################################################
# Script: run_full_pipeline.sh
# Description: Run the complete extension pipeline (both experiments)
#              This is a convenience wrapper that runs all steps sequentially
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
export MODEL="${MODEL:-llama2-7b-chat-hf}"
export PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
export SPARSITY_RATIO="${SPARSITY_RATIO:-0.05}"
export TRAINING_DATA="${TRAINING_DATA:-alpaca_cleaned_no_safety}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export NSAMPLES="${NSAMPLES:-128}"
export SEED="${SEED:-42}"
export DEVICE="${DEVICE:-cuda:0}"
export EVAL_ATTACK="${EVAL_ATTACK:-false}"

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "FULL PIPELINE EXECUTION"
echo "======================================================================"
echo "Model: ${MODEL}"
echo "Prune Method: ${PRUNE_METHOD}"
echo "Sparsity Ratio: ${SPARSITY_RATIO}"
echo "Training Data: ${TRAINING_DATA}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Length: ${MAX_LENGTH}"
echo "Number of Samples: ${NSAMPLES}"
echo "Evaluate Attack: ${EVAL_ATTACK}"
echo "======================================================================"
echo ""
echo "Pipeline stages:"
echo "  1. Neuron identification (safety + utility)"
echo "  2. Experiment 1: Frozen-regime fine-tuning"
echo "  3. Experiment 2: Unfrozen fine-tuning"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# ============================================================================
# Stage 1: Neuron Identification
# ============================================================================
echo ""
echo "======================================================================"
echo "STAGE 1: NEURON IDENTIFICATION"
echo "======================================================================"
echo ""

./scripts/00_setup_neuron_identification.sh

# ============================================================================
# Stage 2: Experiment 1 (Frozen)
# ============================================================================
echo ""
echo "======================================================================"
echo "STAGE 2: EXPERIMENT 1 (FROZEN-REGIME)"
echo "======================================================================"
echo ""

./scripts/01_run_experiment_1_frozen.sh

# ============================================================================
# Stage 3: Experiment 2 (Unfrozen)
# ============================================================================
echo ""
echo "======================================================================"
echo "STAGE 3: EXPERIMENT 2 (UNFROZEN)"
echo "======================================================================"
echo ""

./scripts/02_run_experiment_2_unfrozen.sh

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "FULL PIPELINE COMPLETE"
echo "======================================================================"
echo ""
echo "All results saved to: ./results/${MODEL}/"
echo ""
echo "Directory structure:"
echo "  results/${MODEL}/"
echo "  ├── neuron_identification/"
echo "  │   ├── safety_neurons/"
echo "  │   └── utility_neurons/"
echo "  ├── experiment_1_frozen/"
echo "  │   ├── fine_tuned_model/"
echo "  │   ├── score_dynamics_analysis.json"
if [ "${EVAL_ATTACK}" = "true" ]; then
    echo "  │   └── asr_results_frozen.json"
fi
echo "  └── experiment_2_unfrozen/"
echo "      ├── fine_tuned_model/"
echo "      ├── weight_drift_analysis.json"
if [ "${EVAL_ATTACK}" = "true" ]; then
    echo "      └── asr_results_unfrozen.json"
fi
echo ""
echo "Next steps:"
echo "  - Analyze score_dynamics_analysis.json (Experiment 1)"
echo "  - Analyze weight_drift_analysis.json (Experiment 2)"
if [ "${EVAL_ATTACK}" = "true" ]; then
    echo "  - Compare ASR results between experiments"
fi
echo "======================================================================"
