#!/bin/bash
################################################################################
# Script: 01_run_experiment_1_frozen.sh
# Description: Experiment 1 - Frozen-Regime Fine-Tuning (Wanda Score Dynamics)
#              Tests whether frozen safety neurons become "stranded"
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
MODEL="${MODEL:-llama2-7b-chat-hf}"
PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.05}"
TRAINING_DATA="${TRAINING_DATA:-alpaca_cleaned_no_safety}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_ATTACK="${EVAL_ATTACK:-false}"

# Results directories
RESULTS_BASE="./results/${MODEL}"
NEURON_DIR="${RESULTS_BASE}/neuron_identification"
EXP_DIR="${RESULTS_BASE}/experiment_1_frozen"
MODEL_SAVE_PATH="${EXP_DIR}/fine_tuned_model"

# ============================================================================
# Verify Prerequisites
# ============================================================================
SAFETY_MASKS="${NEURON_DIR}/safety_neurons/original_safety_masks.pt"

if [ ! -f "${SAFETY_MASKS}" ]; then
    echo "ERROR: Safety masks not found at ${SAFETY_MASKS}"
    echo "Please run: ./scripts/00_setup_neuron_identification.sh"
    exit 1
fi

# ============================================================================
# Print Configuration
# ============================================================================
echo "======================================================================"
echo "EXPERIMENT 1: FROZEN-REGIME FINE-TUNING"
echo "======================================================================"
echo "Model: ${MODEL}"
echo "Training Data: ${TRAINING_DATA}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Safety Masks: ${SAFETY_MASKS}"
echo "Results Directory: ${EXP_DIR}"
echo "======================================================================"
echo ""

# ============================================================================
# Step 1: Fine-tune with Frozen Safety Neurons
# ============================================================================
echo "Step 1/2: Fine-tuning with frozen safety-critical neurons..."
echo "----------------------------------------------------------------------"

EVAL_FLAG=""
if [ "${EVAL_ATTACK}" = "true" ]; then
    EVAL_FLAG="--eval_attack"
    echo "Note: ASR evaluation will be performed after fine-tuning"
fi

uv run python main_extension.py \
    --task fine_tune_frozen \
    --model ${MODEL} \
    --safety_masks ${SAFETY_MASKS} \
    --training_data ${TRAINING_DATA} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --prune_method ${PRUNE_METHOD} \
    --prune_data align_short \
    --sparsity_ratio ${SPARSITY_RATIO} \
    --nsamples ${NSAMPLES} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --results_path ${EXP_DIR} \
    ${EVAL_FLAG}

echo ""
echo "✓ Fine-tuning complete. Model saved to: ${MODEL_SAVE_PATH}"
echo ""

# ============================================================================
# Step 2: Analyze Score Dynamics
# ============================================================================
echo "Step 2/2: Analyzing Wanda score dynamics..."
echo "----------------------------------------------------------------------"

uv run python main_extension.py \
    --task eval_score_dynamics \
    --original_safety_masks_path ${NEURON_DIR}/safety_neurons/original_safety_masks.pt \
    --original_safety_scores_path ${NEURON_DIR}/safety_neurons/original_safety_scores.pt \
    --fine_tuned_safety_scores_path ${EXP_DIR}/safety_neurons/fine_tuned_safety_scores.pt \
    --results_path ${EXP_DIR}

echo ""
echo "✓ Score dynamics analysis complete"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "======================================================================"
echo "EXPERIMENT 1 COMPLETE"
echo "======================================================================"
echo ""
echo "Results:"
echo "  Fine-tuned model: ${MODEL_SAVE_PATH}"
echo "  Score analysis:   ${EXP_DIR}/score_dynamics_analysis.json"
if [ "${EVAL_ATTACK}" = "true" ]; then
    echo "  ASR results:      ${EXP_DIR}/asr_results_frozen.json"
fi
echo ""
echo "Interpretation:"
echo "  - Check score_dynamics_analysis.json for 'Overall Score Change'"
echo "  - Large negative change (-10% or more):"
echo "      → Supports Hypothesis A (Representational Drift)"
echo "  - Stable scores (near 0% change):"
echo "      → Supports Hypothesis B (Global Redistribution)"
echo "======================================================================"
