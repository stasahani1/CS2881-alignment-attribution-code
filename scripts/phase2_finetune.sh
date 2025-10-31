#!/bin/bash
# Phase 2: Fine-tune with LoRA and track neuron drift

set -e  # Exit on error

# Configuration
MODEL_NAME="llama2-7b-chat-hf"
MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"  # Use HF identifier to auto-load from cache

# LoRA configuration
LORA_R=8
LORA_ALPHA=16

# Training configuration
NUM_EPOCHS=1
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=1e-4
MAX_LENGTH=512
MAX_STEPS=-1  # -1 for full training, set to smaller number for testing

# Paths
NEURON_GROUPS_DIR="/workspace/CS2881-alignment-attribution-code/neuron_groups"
OUTPUT_DIR="/workspace/CS2881-alignment-attribution-code/finetuned_models"
DRIFT_LOG_DIR="/dev/shm/drift_logs"
INITIAL_WEIGHTS_DIR="/dev/shm/initial_weights"
DRIFT_LOG_INTERVAL=100  # Compute drift every 100 steps

echo "================================================"
echo "Phase 2: Fine-Tuning with LoRA"
echo "================================================"
echo "Model: ${MODEL_NAME}"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "Training: ${NUM_EPOCHS} epochs, lr=${LEARNING_RATE}"
echo "Batch size: ${BATCH_SIZE} (grad accum: ${GRAD_ACCUM})"
echo "Max steps: ${MAX_STEPS}"
echo ""
echo "Output:"
echo "  Model: ${OUTPUT_DIR}"
echo "  Drift logs: ${DRIFT_LOG_DIR}"
echo "  Initial weights: ${INITIAL_WEIGHTS_DIR}"
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DRIFT_LOG_DIR}"
mkdir -p "${INITIAL_WEIGHTS_DIR}"

# Check that neuron groups exist
if [ ! -d "${NEURON_GROUPS_DIR}" ]; then
    echo "Error: Neuron groups directory not found: ${NEURON_GROUPS_DIR}"
    echo "Please run scripts/phase1_identify_neurons.sh first"
    exit 1
fi

echo "Starting fine-tuning..."
python finetune_with_tracking.py \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_length "${MAX_LENGTH}" \
    --max_steps "${MAX_STEPS}" \
    --neuron_groups_dir "${NEURON_GROUPS_DIR}" \
    --drift_log_dir "${DRIFT_LOG_DIR}" \
    --drift_log_interval "${DRIFT_LOG_INTERVAL}" \
    --initial_weights_dir "${INITIAL_WEIGHTS_DIR}"

echo ""
echo "================================================"
echo "Phase 2 Complete!"
echo "================================================"
echo "Fine-tuned model saved to: ${OUTPUT_DIR}"
echo "Drift logs saved to: ${DRIFT_LOG_DIR}"
echo ""
echo "Next step: Run analyze_drift.py to analyze results"
