#!/bin/bash
################################################################################
# Run complete extension pipeline (both experiments)
################################################################################

set -e  # Exit on error

# Default configuration
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
export TOP_P="${TOP_P:-${SPARSITY_RATIO}}"
export TOP_Q="${TOP_Q:-${SPARSITY_RATIO}}"
export SCORE_TMP_DIR="${SCORE_TMP_DIR:-/dev/shm/wanda_scores}"

echo "======================================================================"
echo "FULL PIPELINE: ${MODEL}"
echo "======================================================================"
echo "Sparsity: ${SPARSITY_RATIO} | Epochs: ${NUM_EPOCHS} | Samples: ${NSAMPLES}"
echo "Stages: (1) Neuron ID → (2) Exp 1 Frozen → (3) Exp 2 Unfrozen"
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Run all stages
./scripts/00_setup_neuron_identification.sh
./scripts/01_run_experiment_1_frozen.sh
./scripts/02_run_experiment_2_unfrozen.sh

echo ""
echo "======================================================================"
echo "✓ PIPELINE COMPLETE"
echo "======================================================================"
echo "Results: ./results/${MODEL}/"
echo "  - experiment_1_frozen/score_dynamics_analysis.json"
echo "  - experiment_2_unfrozen/weight_drift_analysis.json"
echo "======================================================================"
