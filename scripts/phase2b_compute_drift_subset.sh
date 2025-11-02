#!/bin/bash
# Phase 2b (subset): Compute neuron weight drift for specific checkpoints
#
# Usage:
#   ./scripts/phase2b_compute_drift_subset.sh checkpoint-500 checkpoint-1000 checkpoint-1500
#
# Or specify full paths:
#   ./scripts/phase2b_compute_drift_subset.sh /workspace/CS2881-alignment-attribution-code/finetuned_models/checkpoint-500

set -e  # Exit on error

# Configuration
MODEL_NAME="llama2-7b-chat-hf"
BASE_MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"  # Use HF identifier to auto-load from cache

# Paths
CHECKPOINT_BASE_DIR="/workspace/CS2881-alignment-attribution-code/finetuned_models"
NEURON_GROUPS_DIR="/workspace/CS2881-alignment-attribution-code/neuron_groups"
INITIAL_WEIGHTS_PATH="/dev/shm/initial_weights/initial_weights.pt"
DRIFT_LOG_DIR="/dev/shm/drift_logs"

echo "================================================"
echo "Phase 2b: Computing Drift for Specific Checkpoints"
echo "================================================"
echo "Model: ${MODEL_NAME}"
echo ""

# Check for checkpoint arguments
if [ $# -eq 0 ]; then
    echo "Error: No checkpoints specified"
    echo ""
    echo "Usage:"
    echo "  $0 checkpoint-500 checkpoint-1000 checkpoint-1500"
    echo ""
    echo "Or specify full paths:"
    echo "  $0 /path/to/checkpoint-500 /path/to/checkpoint-1000"
    exit 1
fi

# Build checkpoint paths
CHECKPOINT_PATHS=()
for arg in "$@"; do
    # Check if it's a full path or just a checkpoint name
    if [[ "$arg" = /* ]]; then
        # Full path
        CHECKPOINT_PATHS+=("$arg")
    else
        # Just checkpoint name, prepend base directory
        CHECKPOINT_PATHS+=("${CHECKPOINT_BASE_DIR}/$arg")
    fi
done

echo "Checkpoints to process:"
for checkpoint in "${CHECKPOINT_PATHS[@]}"; do
    echo "  - $checkpoint"
done
echo ""

echo "Input:"
echo "  Neuron groups: ${NEURON_GROUPS_DIR}"
echo "  Initial weights: ${INITIAL_WEIGHTS_PATH}"
echo ""
echo "Output:"
echo "  Drift logs: ${DRIFT_LOG_DIR}"
echo ""

# Check that initial weights exist
if [ ! -f "${INITIAL_WEIGHTS_PATH}" ]; then
    echo "Error: Initial weights not found: ${INITIAL_WEIGHTS_PATH}"
    echo "Please run scripts/phase2_finetune.sh first"
    exit 1
fi

# Check that neuron groups exist
if [ ! -d "${NEURON_GROUPS_DIR}" ]; then
    echo "Error: Neuron groups directory not found: ${NEURON_GROUPS_DIR}"
    echo "Please run scripts/phase1_identify_neurons.sh first"
    exit 1
fi

# Create output directory
mkdir -p "${DRIFT_LOG_DIR}"

echo "Starting drift computation..."
python compute_drift.py \
    --base_model_path "${BASE_MODEL_PATH}" \
    --checkpoint_paths "${CHECKPOINT_PATHS[@]}" \
    --neuron_groups_dir "${NEURON_GROUPS_DIR}" \
    --initial_weights_path "${INITIAL_WEIGHTS_PATH}" \
    --drift_log_dir "${DRIFT_LOG_DIR}"

echo ""
echo "================================================"
echo "Phase 2b Complete!"
echo "================================================"
echo "Drift logs saved to: ${DRIFT_LOG_DIR}"
echo ""
echo "Next step: Run analyze_drift.py to analyze results"
