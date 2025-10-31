#!/bin/bash
# Master script to run the complete neuron drift experiment
# This runs all phases sequentially

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================================================"
echo "SAFETY-CRITICAL NEURON DRIFT EXPERIMENT"
echo "========================================================================"
echo ""
echo "This script will run the complete experiment pipeline:"
echo "  Phase 1a: Compute SNIP scores (safety + utility datasets)"
echo "  Phase 1b: Identify neuron groups (4 groups)"
echo "  Phase 2:  Fine-tune with LoRA and track drift"
echo "  Phase 3:  Analyze drift patterns and generate report"
echo ""
echo "========================================================================"
echo ""

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
echo "========================================================================"
echo "PHASE 1A: Computing SNIP Scores"
echo "========================================================================"
bash "${SCRIPT_DIR}/phase1_compute_snip_scores.sh"

echo ""
echo "========================================================================"
echo "PHASE 1B: Identifying Neuron Groups"
echo "========================================================================"
bash "${SCRIPT_DIR}/phase1_identify_neurons.sh"

echo ""
echo "========================================================================"
echo "PHASE 2: Fine-Tuning with LoRA"
echo "========================================================================"
bash "${SCRIPT_DIR}/phase2_finetune.sh"

echo ""
echo "========================================================================"
echo "PHASE 3: Analyzing Drift Patterns"
echo "========================================================================"
cd /workspace/CS2881-alignment-attribution-code
python analyze_drift.py \
    --drift_log_dir /dev/shm/drift_logs \
    --output_dir /workspace/CS2881-alignment-attribution-code/results

echo ""
echo "========================================================================"
echo "EXPERIMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "Results available at:"
echo "  - Analysis report: /workspace/CS2881-alignment-attribution-code/results/drift_analysis_report.txt"
echo "  - Figures: /workspace/CS2881-alignment-attribution-code/results/figures/"
echo "  - Statistical results: /workspace/CS2881-alignment-attribution-code/results/statistical_results.json"
echo ""
echo "Model and data:"
echo "  - Fine-tuned model: /workspace/CS2881-alignment-attribution-code/finetuned_models/"
echo "  - Neuron groups: /workspace/CS2881-alignment-attribution-code/neuron_groups/"
echo ""
echo "Temporary files (can be cleaned up):"
echo "  - SNIP scores: /dev/shm/snip_scores/"
echo "  - Drift logs: /dev/shm/drift_logs/"
echo "  - Initial weights: /dev/shm/initial_weights/"
echo ""
