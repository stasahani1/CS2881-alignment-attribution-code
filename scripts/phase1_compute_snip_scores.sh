#!/bin/bash
# Phase 1: Compute SNIP scores for safety and utility datasets
# This script dumps SNIP scores to /dev/shm for fast access

set -e  # Exit on error

# Configuration
MODEL="llama2-7b-chat-hf"
CACHE_DIR="models/llama-2-7b-chat-hf"  # Use pre-downloaded models
METHOD="wandg"  # SNIP method
SPARSITY_RATIO=0.01  # Not used for scoring, but required parameter
NSAMPLES=128  # Number of calibration samples
SEED=0

# Output directories (using /dev/shm for large temporary files)
# SCORE_BASE_DIR="/dev/shm/snip_scores"
SCORE_BASE_DIR="/tmp/snip_scores"
SAFETY_DATASET="align_short"
UTILITY_DATASET="alpaca_cleaned_no_safety"
SAFETY_SCORE_DIR="${SCORE_BASE_DIR}/${SAFETY_DATASET}"
UTILITY_SCORE_DIR="${SCORE_BASE_DIR}/${UTILITY_DATASET}"

echo "================================================"
echo "Phase 1: Computing SNIP Scores"
echo "================================================"
echo "Model: ${MODEL}"
echo "Method: ${METHOD}"
echo "Samples: ${NSAMPLES}"
echo ""

# Create output directories
mkdir -p "${SAFETY_SCORE_DIR}"
mkdir -p "${UTILITY_SCORE_DIR}"

echo "Step 1/2: Computing SNIP scores on safety dataset (align)..."
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
echo "Step 2/2: Computing SNIP scores on utility dataset (alpaca_cleaned_no_safety)..."
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
echo "Phase 1 Complete!"
echo "================================================"
echo "SNIP scores saved to:"
echo "  Safety:  ${SAFETY_SCORE_DIR}/snip_score/"
echo "  Utility: ${UTILITY_SCORE_DIR}/snip_score/"
echo ""
echo "Next step: Run phase1_identify_neurons.sh"
