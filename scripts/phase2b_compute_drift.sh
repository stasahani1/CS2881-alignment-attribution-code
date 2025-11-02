#!/bin/bash
# Phase 2b: Compute neuron weight drift for all checkpoints

set -e  # Exit on error

# Configuration
MODEL_NAME="llama2-7b-chat-hf"
BASE_MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"  # Use HF identifier to auto-load from cache

# Paths
CHECKPOINT_DIR="/workspace/CS2881-alignment-attribution-code/finetuned_models"
NEURON_GROUPS_DIR="/workspace/CS2881-alignment-attribution-code/neuron_groups"
INITIAL_WEIGHTS_PATH="/dev/shm/initial_weights/initial_weights.pt"
DRIFT_LOG_DIR="/dev/shm/drift_logs"

echo "================================================"
echo "Phase 2b: Computing Drift for All Checkpoints"
echo "================================================"
echo "Model: ${MODEL_NAME}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo ""
echo "Input:"
echo "  Neuron groups: ${NEURON_GROUPS_DIR}"
echo "  Initial weights: ${INITIAL_WEIGHTS_PATH}"
echo ""
echo "Output:"
echo "  Drift logs: ${DRIFT_LOG_DIR}"
echo ""

# Check that checkpoint directory exists
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: Checkpoint directory not found: ${CHECKPOINT_DIR}"
    echo "Please run scripts/phase2_finetune.sh first"
    exit 1
fi

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
    --checkpoint_dir "${CHECKPOINT_DIR}" \
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
