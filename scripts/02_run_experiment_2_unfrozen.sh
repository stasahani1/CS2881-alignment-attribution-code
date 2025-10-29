#!/bin/bash
################################################################################
# Script: 02_run_experiment_2_unfrozen.sh
# Description: Experiment 2 - Unfrozen Fine-Tuning (Safety Neuron Drift)
#              Tests whether safety neurons are more fragile than others
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
MODEL="${MODEL:-llama2-7b-chat-hf}"
TRAINING_DATA="${TRAINING_DATA:-alpaca_cleaned_no_safety}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_ATTACK="${EVAL_ATTACK:-false}"

# Results directories
RESULTS_BASE="./results/${MODEL}"
NEURON_DIR="${RESULTS_BASE}/neuron_identification"
EXP_DIR="${RESULTS_BASE}/experiment_2_unfrozen"
MODEL_SAVE_PATH="${EXP_DIR}/fine_tuned_model"

# ============================================================================
# Verify Prerequisites
# ============================================================================
SAFETY_MASKS="${NEURON_DIR}/safety_neurons/original_safety_masks.pt"
UTILITY_MASKS="${NEURON_DIR}/utility_neurons/original_utility_masks.pt"

if [ ! -f "${SAFETY_MASKS}" ]; then
    echo "ERROR: Safety masks not found at ${SAFETY_MASKS}"
    echo "Please run: ./scripts/00_setup_neuron_identification.sh"
    exit 1
fi

if [ ! -f "${UTILITY_MASKS}" ]; then
    echo "ERROR: Utility masks not found at ${UTILITY_MASKS}"
    echo "Please run: ./scripts/00_setup_neuron_identification.sh"
    exit 1
fi

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "EXPERIMENT 2: UNFROZEN FINE-TUNING (WEIGHT DRIFT)"
echo "======================================================================"
echo "Model: ${MODEL}"
echo "Training Data: ${TRAINING_DATA}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Safety Masks: ${SAFETY_MASKS}"
echo "Utility Masks: ${UTILITY_MASKS}"
echo "Results Directory: ${EXP_DIR}"
echo "======================================================================"
echo ""

# ============================================================================
# Step 1: Fine-tune without Freezing (Track Weight Drift)
# ============================================================================
echo "Step 1/2: Fine-tuning without freezing (capturing weight drift)..."
echo "----------------------------------------------------------------------"

EVAL_FLAG=""
if [ "${EVAL_ATTACK}" = "true" ]; then
    EVAL_FLAG="--eval_attack"
    echo "Note: ASR evaluation will be performed after fine-tuning"
fi

uv run python main_extension.py \
    --task fine_tune_unfrozen \
    --model ${MODEL} \
    --training_data ${TRAINING_DATA} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --results_path ${EXP_DIR} \
    ${EVAL_FLAG}

echo ""
echo "✓ Fine-tuning complete. Model saved to: ${MODEL_SAVE_PATH}"
echo ""

# ============================================================================
# Step 2: Analyze Weight Drift
# ============================================================================
echo "Step 2/2: Analyzing weight drift by neuron category..."
echo "----------------------------------------------------------------------"

uv run python main_extension.py \
    --task eval_weight_drift \
    --fine_tuned_model_path ${MODEL_SAVE_PATH} \
    --safety_masks_path ${SAFETY_MASKS} \
    --utility_masks_path ${UTILITY_MASKS} \
    --original_weights_path ${EXP_DIR}/original_weights.pt \
    --results_path ${EXP_DIR}

echo ""
echo "✓ Weight drift analysis complete"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "======================================================================"
echo "EXPERIMENT 2 COMPLETE"
echo "======================================================================"
echo ""
echo "Results:"
echo "  Fine-tuned model: ${MODEL_SAVE_PATH}"
echo "  Drift analysis:   ${EXP_DIR}/weight_drift_analysis.json"
if [ "${EVAL_ATTACK}" = "true" ]; then
    echo "  ASR results:      ${EXP_DIR}/asr_results_unfrozen.json"
fi
echo ""
echo "Interpretation:"
echo "  - Check weight_drift_analysis.json for cosine similarity comparison"
echo "  - Safety < Random (lower cosine similarity):"
echo "      → Supports Hypothesis C (Fragile Safety Neurons)"
echo "  - Safety ≈ Random (similar cosine similarity):"
echo "      → Supports Hypothesis D (Recontextualization)"
echo "======================================================================"
