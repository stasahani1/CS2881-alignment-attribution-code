#!/bin/bash
# Master script to run the complete neuron overlap experiment
# This runs all phases sequentially

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Confirm before running
read -p "This will take several hours. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting experiment..."
echo ""

# Phase 1a: Compute SNIP scores
bash "${SCRIPT_DIR}/phase1_compute_snip_scores.sh"
bash "${SCRIPT_DIR}/phase1_identify_neurons.sh"
bash "${SCRIPT_DIR}/phase2_finetune.sh"
bash "${SCRIPT_DIR}/phase3_get_set_diff_for_finetuned_model.sh"

cd /workspace/CS2881-alignment-attribution-code
source .venv/bin/activate
uv run python eval_results.py \
    --original_neuron_groups_dir "outputs/neuron_groups" \
    --finetuned_neuron_groups_dir "outputs/neuron_groups_finetuned" \
    --output_dir "outputs/analysis"



