#!/bin/bash
set -euo pipefail

echo "Quick test: Experiment 2 (set-diff IDs, 1 epoch)"
read -p "Press Enter to continue or Ctrl+C to abort... "

export MODEL="${MODEL:-llama2-7b-chat-hf}"
export PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
export SPARSITY_RATIO="${SPARSITY_RATIO:-0.01}"
export TOP_P="${TOP_P:-${SPARSITY_RATIO}}"
export TOP_Q="${TOP_Q:-${SPARSITY_RATIO}}"
export NSAMPLES="${NSAMPLES:-32}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export TRAINING_DATA="${TRAINING_DATA:-alpaca_cleaned_no_safety}"
export EVAL_ATTACK="false"

./scripts/00_setup_neuron_identification.sh
./scripts/02_run_experiment_2_unfrozen.sh

echo "Quick test complete → ./results/${MODEL}/experiment_2_unfrozen/"
