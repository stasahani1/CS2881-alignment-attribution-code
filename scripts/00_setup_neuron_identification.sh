#!/bin/bash
################################################################################
# Script: 00_setup_neuron_identification.sh
# Description: Identify safety-critical and utility-critical neurons
#              This only needs to be run ONCE per model
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
MODEL="${MODEL:-llama2-7b-chat-hf}"
PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.05}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"

# Results directory
RESULTS_BASE="./results/${MODEL}"
NEURON_DIR="${RESULTS_BASE}/neuron_identification"

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "NEURON IDENTIFICATION SETUP"
echo "======================================================================"
echo "Model: ${MODEL}"
echo "Method: ${PRUNE_METHOD}"
echo "Sparsity Ratio: ${SPARSITY_RATIO}"
echo "Number of Samples: ${NSAMPLES}"
echo "Results Directory: ${NEURON_DIR}"
echo "======================================================================"
echo ""

# ============================================================================
# Step 1: Identify Safety-Critical Neurons
# ============================================================================
echo "Step 1/2: Identifying safety-critical neurons..."
echo "----------------------------------------------------------------------"

uv run python main_extension.py \
    --task identify_safety_neurons \
    --model ${MODEL} \
    --prune_method ${PRUNE_METHOD} \
    --prune_data align_short \
    --sparsity_ratio ${SPARSITY_RATIO} \
    --nsamples ${NSAMPLES} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --results_path ${NEURON_DIR}

echo ""
echo "✓ Safety-critical neurons identified and saved to:"
echo "  ${NEURON_DIR}/safety_neurons/"
echo ""

# ============================================================================
# Step 2: Identify Utility-Critical Neurons
# ============================================================================
echo "Step 2/2: Identifying utility-critical neurons..."
echo "----------------------------------------------------------------------"

uv run python main_extension.py \
    --task identify_utility_neurons \
    --model ${MODEL} \
    --prune_method ${PRUNE_METHOD} \
    --prune_data alpaca_cleaned_no_safety \
    --sparsity_ratio ${SPARSITY_RATIO} \
    --nsamples ${NSAMPLES} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --results_path ${NEURON_DIR}

echo ""
echo "✓ Utility-critical neurons identified and saved to:"
echo "  ${NEURON_DIR}/utility_neurons/"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "======================================================================"
echo "NEURON IDENTIFICATION COMPLETE"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  Safety masks:  ${NEURON_DIR}/safety_neurons/original_safety_masks.pt"
echo "  Safety scores: ${NEURON_DIR}/safety_neurons/original_safety_scores.pt"
echo "  Utility masks: ${NEURON_DIR}/utility_neurons/original_utility_masks.pt"
echo "  Utility scores: ${NEURON_DIR}/utility_neurons/original_utility_scores.pt"
echo ""
echo "Next steps:"
echo "  - Run Experiment 1: ./scripts/01_run_experiment_1_frozen.sh"
echo "  - Run Experiment 2: ./scripts/02_run_experiment_2_unfrozen.sh"
echo "======================================================================"
