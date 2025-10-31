#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-llama2-7b-chat-hf}"
TRAINING_DATA="${TRAINING_DATA:-alpaca_cleaned_no_safety}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_ATTACK="${EVAL_ATTACK:-false}"

RESULTS_BASE="./results/${MODEL}"
NEURON_DIR="${RESULTS_BASE}/neuron_identification"
EXP_DIR="${RESULTS_BASE}/experiment_2_unfrozen"
MODEL_SAVE_PATH="/dev/shm/fine_tuned_models/${MODEL}/experiment_2_unfrozen"
SAFETY_MASKS="${NEURON_DIR}/safety_neurons/original_safety_masks.pt"
UTILITY_MASKS="${NEURON_DIR}/utility_neurons/original_utility_masks.pt"

[ -f "${SAFETY_MASKS}" ] || { echo "missing safety masks: ${SAFETY_MASKS}"; exit 1; }
[ -f "${UTILITY_MASKS}" ] || { echo "missing utility masks: ${UTILITY_MASKS}"; exit 1; }

mkdir -p "${EXP_DIR}"

echo "[Exp2] fine-tune without freezing"
cmd_ft=(uv run python main_extension.py
  --task fine_tune_unfrozen
  --model "${MODEL}"
  --training_data "${TRAINING_DATA}"
  --num_epochs "${NUM_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --model_save_path "${MODEL_SAVE_PATH}"
  --results_path "${EXP_DIR}")

if [ "${EVAL_ATTACK}" = "true" ]; then
  cmd_ft+=(--eval_attack)
fi

"${cmd_ft[@]}"

echo "[Exp2] evaluate weight drift"
uv run python main_extension.py \
  --task eval_weight_drift \
  --fine_tuned_model_path "${MODEL_SAVE_PATH}" \
  --safety_masks_path "${SAFETY_MASKS}" \
  --utility_masks_path "${UTILITY_MASKS}" \
  --original_weights_path "${EXP_DIR}/original_weights.pt" \
  --results_path "${EXP_DIR}"

echo "Experiment 2 outputs → ${EXP_DIR}"
