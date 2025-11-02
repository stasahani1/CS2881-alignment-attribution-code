#!/bin/bash
# Phase 1b: Identify neuron groups from SNIP scores
# Extracts safety-critical, utility-critical, and random neuron sets

set -e  # Exit on error

# Configuration
SCORE_BASE_DIR="/tmp/snip_scores"
OUTPUT_DIR="outputs/neuron_groups"
SAFETY_DATASET="align_short"
UTILITY_DATASET="alpaca_cleaned_no_safety"
# Neuron identification parameters
SNIP_TOP_K=0.01          # Top 1% for SNIP method
SET_DIFF_P=0.03           # Top-p% utility neurons to exclude
SET_DIFF_Q=0.03           # Top-q% safety neurons to consider
# Methods to run
METHODS="top_safety set_difference"

echo "================================================"
echo "Phase 1b: Identifying Neuron Groups"
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
    --methods ${METHODS}
    --seed 0

echo ""
echo "================================================"
echo "Phase 1b Complete!"
echo "================================================"
echo "Neuron groups saved to: ${OUTPUT_DIR}/"
echo "  - neuron_groups_snip_top.json"
echo "  - neuron_groups_set_diff.json"
echo "  - neuron_groups_utility.json"
echo "  - neuron_groups_random.json"
echo ""
echo "Next step: Run scripts/phase2_finetune.sh"
