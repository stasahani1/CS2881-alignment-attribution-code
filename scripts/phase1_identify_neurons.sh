#!/bin/bash
# Phase 1b: Identify neuron groups from SNIP scores
# Extracts safety-critical, utility-critical, and random neuron sets

set -e  # Exit on error

# Configuration
SCORE_BASE_DIR="/dev/shm/snip_scores"
OUTPUT_DIR="/workspace/CS2881-alignment-attribution-code/neuron_groups"

# Neuron identification parameters
# Optimized values: reduced from 1% to 0.1% for 10x speedup
# Still statistically valid (~6.5M neurons) and focuses on most critical neurons
SNIP_TOP_K=0.001          # Top 0.1% for SNIP method (was 0.01)
SET_DIFF_P=0.05           # Top 5% utility neurons to exclude (was 0.05)
SET_DIFF_Q=0.002          # Top 0.2% safety neurons to consider (was 0.02)

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
    --output_dir "${OUTPUT_DIR}" \
    --snip_top_k "${SNIP_TOP_K}" \
    --set_diff_p "${SET_DIFF_P}" \
    --set_diff_q "${SET_DIFF_Q}" \
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
