#!/bin/bash
################################################################################
# Script: check_status.sh
# Description: Check which experiments have been completed for each model
################################################################################

# ============================================================================
# Configuration
# ============================================================================
RESULTS_DIR="./results"

# ============================================================================
# Helper Functions
# ============================================================================
check_file() {
    if [ -f "$1" ]; then
        echo "✓"
    else
        echo "✗"
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "✓"
    else
        echo "✗"
    fi
}

# ============================================================================
# Main Status Check
# ============================================================================
echo "======================================================================"
echo "EXPERIMENT STATUS CHECKER"
echo "======================================================================"
echo ""

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "No results directory found. No experiments have been run yet."
    echo ""
    echo "To get started, run:"
    echo "  ./scripts/quick_test.sh        # Quick test"
    echo "  ./scripts/run_full_pipeline.sh # Full pipeline"
    exit 0
fi

# Find all model directories
MODELS=$(find ${RESULTS_DIR} -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

if [ -z "${MODELS}" ]; then
    echo "No model results found in ${RESULTS_DIR}/"
    exit 0
fi

for MODEL in ${MODELS}; do
    MODEL_DIR="${RESULTS_DIR}/${MODEL}"

    echo "----------------------------------------------------------------------"
    echo "Model: ${MODEL}"
    echo "----------------------------------------------------------------------"
    echo ""

    # Check neuron identification
    echo "Neuron Identification:"
    SAFETY_MASKS="${MODEL_DIR}/neuron_identification/safety_neurons/original_safety_masks.pt"
    UTILITY_MASKS="${MODEL_DIR}/neuron_identification/utility_neurons/original_utility_masks.pt"

    echo "  Safety neurons:  $(check_file ${SAFETY_MASKS})"
    echo "  Utility neurons: $(check_file ${UTILITY_MASKS})"
    echo ""

    # Check Experiment 1
    echo "Experiment 1 (Frozen-Regime):"
    EXP1_MODEL="${MODEL_DIR}/experiment_1_frozen/fine_tuned_model"
    EXP1_SCORES="${MODEL_DIR}/experiment_1_frozen/safety_neurons/fine_tuned_safety_scores.pt"
    EXP1_ANALYSIS="${MODEL_DIR}/experiment_1_frozen/score_dynamics_analysis.json"
    EXP1_ASR="${MODEL_DIR}/experiment_1_frozen/asr_results_frozen.json"

    echo "  Fine-tuned model:    $(check_dir ${EXP1_MODEL})"
    echo "  Score analysis:      $(check_file ${EXP1_ANALYSIS})"
    echo "  ASR evaluation:      $(check_file ${EXP1_ASR})"

    if [ -f "${EXP1_ANALYSIS}" ]; then
        # Extract key results if jq is available
        if command -v jq &> /dev/null; then
            ORIG_MEAN=$(jq -r '.score_distributions.original_safety.mean // "N/A"' ${EXP1_ANALYSIS})
            NEW_MEAN=$(jq -r '.score_distributions.new_safety.mean // "N/A"' ${EXP1_ANALYSIS})

            if [ "${ORIG_MEAN}" != "N/A" ] && [ "${NEW_MEAN}" != "N/A" ]; then
                PCT_CHANGE=$(python3 -c "print(f'{((${NEW_MEAN} - ${ORIG_MEAN}) / ${ORIG_MEAN} * 100):+.2f}')" 2>/dev/null || echo "N/A")
                echo "  Score change:        ${PCT_CHANGE}%"

                if command -v bc &> /dev/null; then
                    IS_NEGATIVE=$(echo "${PCT_CHANGE} < -10" | bc 2>/dev/null)
                    if [ "${IS_NEGATIVE}" = "1" ]; then
                        echo "  Interpretation:      Hypothesis A (Representational Drift)"
                    else
                        echo "  Interpretation:      Hypothesis B (Global Redistribution)"
                    fi
                fi
            fi
        fi
    fi
    echo ""

    # Check Experiment 2
    echo "Experiment 2 (Unfrozen):"
    EXP2_MODEL="${MODEL_DIR}/experiment_2_unfrozen/fine_tuned_model"
    EXP2_WEIGHTS="${MODEL_DIR}/experiment_2_unfrozen/original_weights.pt"
    EXP2_ANALYSIS="${MODEL_DIR}/experiment_2_unfrozen/weight_drift_analysis.json"
    EXP2_ASR="${MODEL_DIR}/experiment_2_unfrozen/asr_results_unfrozen.json"

    echo "  Fine-tuned model:    $(check_dir ${EXP2_MODEL})"
    echo "  Original weights:    $(check_file ${EXP2_WEIGHTS})"
    echo "  Drift analysis:      $(check_file ${EXP2_ANALYSIS})"
    echo "  ASR evaluation:      $(check_file ${EXP2_ASR})"

    if [ -f "${EXP2_ANALYSIS}" ]; then
        # Extract key results if jq is available
        if command -v jq &> /dev/null; then
            SAFETY_COS=$(jq -r '.safety.cosine_similarity.mean // "N/A"' ${EXP2_ANALYSIS})
            RANDOM_COS=$(jq -r '.random.cosine_similarity.mean // "N/A"' ${EXP2_ANALYSIS})

            if [ "${SAFETY_COS}" != "N/A" ] && [ "${RANDOM_COS}" != "N/A" ]; then
                echo "  Safety cosine sim:   ${SAFETY_COS}"
                echo "  Random cosine sim:   ${RANDOM_COS}"

                if command -v bc &> /dev/null; then
                    DIFF=$(echo "${SAFETY_COS} - ${RANDOM_COS}" | bc 2>/dev/null)
                    IS_LESS=$(echo "${DIFF} < -0.05" | bc 2>/dev/null)
                    if [ "${IS_LESS}" = "1" ]; then
                        echo "  Interpretation:      Hypothesis C (Fragile Safety Neurons)"
                    else
                        echo "  Interpretation:      Hypothesis D (Recontextualization)"
                    fi
                fi
            fi
        fi
    fi
    echo ""
done

echo "======================================================================"
echo ""
echo "Legend:"
echo "  ✓ = Complete"
echo "  ✗ = Not done yet"
echo ""
echo "To run missing experiments:"
echo "  ./scripts/00_setup_neuron_identification.sh  # If neurons not identified"
echo "  ./scripts/01_run_experiment_1_frozen.sh      # For Experiment 1"
echo "  ./scripts/02_run_experiment_2_unfrozen.sh    # For Experiment 2"
echo "  ./scripts/run_full_pipeline.sh               # For complete pipeline"
echo "======================================================================"
