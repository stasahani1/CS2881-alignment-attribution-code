#!/bin/bash
################################################################################
# Check experiment completion status
################################################################################

RESULTS_DIR="./results"

check_exists() {
    [ -e "$1" ] && echo "✓" || echo "✗"
}

echo "======================================================================"
echo "EXPERIMENT STATUS"
echo "======================================================================"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "No results found. Run: ./scripts/quick_test.sh"
    exit 0
fi

for MODEL in $(find ${RESULTS_DIR} -mindepth 1 -maxdepth 1 -type d -exec basename {} \;); do
    MODEL_DIR="${RESULTS_DIR}/${MODEL}"

    echo ""
    echo "Model: ${MODEL}"
    echo "----------------------------------------------------------------------"

    # Check completion
    SAFETY_MASKS="${MODEL_DIR}/neuron_identification/safety_neurons/original_safety_masks.pt"
    UTILITY_MASKS="${MODEL_DIR}/neuron_identification/utility_neurons/original_utility_masks.pt"
    EXP1_ANALYSIS="${MODEL_DIR}/experiment_1_frozen/score_dynamics_analysis.json"
    EXP2_ANALYSIS="${MODEL_DIR}/experiment_2_unfrozen/weight_drift_analysis.json"

    echo "Setup:        $(check_exists ${SAFETY_MASKS}) Safety  $(check_exists ${UTILITY_MASKS}) Utility"
    echo "Experiment 1: $(check_exists ${EXP1_ANALYSIS}) Frozen-regime (Wanda score dynamics)"
    echo "Experiment 2: $(check_exists ${EXP2_ANALYSIS}) Unfrozen (Weight drift)"
done

echo ""
echo "======================================================================"
echo "Legend: ✓ = Done, ✗ = Not done"
echo ""
echo "Run experiments:"
echo "  ./scripts/quick_test.sh                         # Quick test"
echo "  ./scripts/01_run_experiment_1_frozen.sh         # Experiment 1"
echo "  ./scripts/02_run_experiment_2_unfrozen.sh       # Experiment 2"
echo "  ./scripts/run_full_pipeline.sh                  # Full pipeline"
echo "======================================================================"
