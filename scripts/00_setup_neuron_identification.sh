#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-llama2-7b-chat-hf}"
PRUNE_METHOD="${PRUNE_METHOD:-wanda}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
SCORE_TMP_DIR="${SCORE_TMP_DIR:-/dev/shm/wanda_scores}"
DEFAULT_RATIO="${SPARSITY_RATIO:-0.05}"
TOP_P="${TOP_P:-${DEFAULT_RATIO}}"
TOP_Q="${TOP_Q:-${DEFAULT_RATIO}}"

RESULTS_BASE="./results/${MODEL}"
NEURON_DIR="${RESULTS_BASE}/neuron_identification"

mkdir -p "${NEURON_DIR}"

echo "[SetDiff] Identifying safety neurons → ${NEURON_DIR}"
uv run python main_extension.py \
    --task identify_safety_neurons \
    --model "${MODEL}" \
    --prune_method "${PRUNE_METHOD}" \
    --selection_strategy set_difference \
    --prune_data align_short \
    --utility_prune_data alpaca_cleaned_no_safety \
    --top_p "${TOP_P}" \
    --top_q "${TOP_Q}" \
    --nsamples "${NSAMPLES}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --score_tmp_dir "${SCORE_TMP_DIR}" \
    --results_path "${NEURON_DIR}"

echo "[SetDiff] Identifying utility neurons → ${NEURON_DIR}"
uv run python main_extension.py \
    --task identify_utility_neurons \
    --model "${MODEL}" \
    --prune_method "${PRUNE_METHOD}" \
    --selection_strategy set_difference \
    --prune_data alpaca_cleaned_no_safety \
    --utility_prune_data align_short \
    --top_p "${TOP_P}" \
    --top_q "${TOP_Q}" \
    --nsamples "${NSAMPLES}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --score_tmp_dir "${SCORE_TMP_DIR}" \
    --results_path "${NEURON_DIR}"

echo "Neuron masks & metadata saved under ${NEURON_DIR}"
