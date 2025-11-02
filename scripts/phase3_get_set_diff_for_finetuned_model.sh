#!/bin/bash
# Phase 1: Compute SNIP scores for safety and utility datasets

set -e  # Exit on error

# Configuration
MODEL="models/finetuned_models"
CACHE_DIR="models/huggingface/hub"  # Use pre-downloaded models
METHOD="wandg"  # SNIP method
SPARSITY_RATIO=0.01  # Not used for scoring, but required parameter
NSAMPLES=128  # Number of calibration samples
SEED=0

# Output directories
SCORE_BASE_DIR="outputs/snip_scores_finetuned"
OUTPUT_DIR="outputs/neuron_groups_finetuned"
SAFETY_DATASET="align_short"
UTILITY_DATASET="alpaca_cleaned_no_safety"
SAFETY_SCORE_DIR="${SCORE_BASE_DIR}/${SAFETY_DATASET}"
UTILITY_SCORE_DIR="${SCORE_BASE_DIR}/${UTILITY_DATASET}"

# Neuron identification parameters
SNIP_TOP_K=0.01          # Top 1% for SNIP method
SET_DIFF_P=0.03           # Top-p% utility neurons to exclude
SET_DIFF_Q=0.03           # Top-q% safety neurons to consider
# Methods to run
METHODS="top_safety set_difference"

echo "================================================"
echo "Phase 3a: Computing SNIP Scores for finetuned models"
echo "================================================"
echo "Model: ${MODEL}"
echo "Method: ${METHOD}"
echo "Samples: ${NSAMPLES}"
echo ""

# Create output directories
mkdir -p "${SAFETY_SCORE_DIR}"
mkdir -p "${UTILITY_SCORE_DIR}"

echo "Step 1/2: Computing SNIP scores on safety dataset (${SAFETY_DATASET})..."
echo "Output: ${SAFETY_SCORE_DIR}"
python main.py \
    --model "${MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --prune_method "${METHOD}" \
    --prune_data "${SAFETY_DATASET}" \
    --sparsity_ratio "${SPARSITY_RATIO}" \
    --save "${SAFETY_SCORE_DIR}" \
    --dump_wanda_score \
    --nsamples "${NSAMPLES}" \
    --seed "${SEED}"

echo ""
echo "Step 2/2: Computing SNIP scores on utility dataset (${UTILITY_DATASET})..."
echo "Output: ${UTILITY_SCORE_DIR}"
python main.py \
    --model "${MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --prune_method "${METHOD}" \
    --prune_data "${UTILITY_DATASET}" \
    --sparsity_ratio "${SPARSITY_RATIO}" \
    --save "${UTILITY_SCORE_DIR}" \
    --dump_wanda_score \
    --nsamples "${NSAMPLES}" \
    --seed "${SEED}"

echo ""
echo "================================================"
echo "Phase 3a Complete!"
echo "================================================"
echo "SNIP scores saved to:"
echo "  Safety:  ${SAFETY_SCORE_DIR}/snip_score/"
echo "  Utility: ${UTILITY_SCORE_DIR}/snip_score/"
echo ""


echo "================================================"
echo "Phase 3b: Identifying Neuron Groups"
echo "================================================"
echo "Methods: SNIP top-k, Set Difference"
echo "SNIP top-k: ${SNIP_TOP_K}"
echo "Set difference: p=${SET_DIFF_P}, q=${SET_DIFF_Q}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Identifying neuron groups..."
python identify_neuron_groups.py \
    --score_base_dir "${SCORE_BASE_DIR}" \
    --safety_dataset "${SAFETY_DATASET}" \
    --utility_dataset "${UTILITY_DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --snip_top_k "${SNIP_TOP_K}" \
    --set_diff_p "${SET_DIFF_P}" \
    --set_diff_q "${SET_DIFF_Q}" \
    --methods ${METHODS} \
    --seed 0

echo ""
echo "================================================"
echo "Phase 3b Complete!"
echo "================================================"
echo "Neuron groups saved to: ${OUTPUT_DIR}/"
echo "  - neuron_groups_snip_top.json"
echo "  - neuron_groups_set_diff.json"
echo "  - neuron_groups_utility.json"
echo "  - neuron_groups_random.json"
echo ""

