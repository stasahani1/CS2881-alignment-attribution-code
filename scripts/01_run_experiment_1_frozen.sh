#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-llama2-7b-chat-hf}"
PRUNE_METHOD="${PRUNE_METHOD:-wandg}"
SPARSITY_RATIO="${SPARSITY_RATIO:-0.05}"
NSAMPLES="${NSAMPLES:-128}"
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
EXP_DIR="${RESULTS_BASE}/experiment_1_frozen"
MODEL_SAVE_PATH="/dev/shm/fine_tuned_models/${MODEL}/experiment_1_frozen"
SAFETY_MASKS="${NEURON_DIR}/safety_neurons/original_safety_masks.pt"

[ -f "${SAFETY_MASKS}" ] || { echo "missing safety masks: ${SAFETY_MASKS}"; exit 1; }

mkdir -p "${EXP_DIR}"

echo "[Exp1] fine-tune with frozen safety neurons"
cmd=(uv run python main_extension.py
  --task fine_tune_frozen
  --model "${MODEL}"
  --safety_masks "${SAFETY_MASKS}"
  --training_data "${TRAINING_DATA}"
  --num_epochs "${NUM_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --prune_method "${PRUNE_METHOD}"
  --prune_data align_short
  --sparsity_ratio "${SPARSITY_RATIO}"
  --nsamples "${NSAMPLES}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --model_save_path "${MODEL_SAVE_PATH}"
  --results_path "${EXP_DIR}")

if [ "${EVAL_ATTACK}" = "true" ]; then
  cmd+=(--eval_attack)
fi

if [ "${USE_PEFT:-false}" = "true" ]; then
  cmd+=(--use_peft --peft_r "${PEFT_R:-8}" --peft_alpha "${PEFT_ALPHA:-16}" --peft_target_modules "${PEFT_TARGET_MODULES:-q_proj,v_proj}")
fi

"${cmd[@]}"

echo "[Exp1] analyze score dynamics"
uv run python main_extension.py \
  --task eval_score_dynamics \
  --original_safety_masks_path "${NEURON_DIR}/safety_neurons/original_safety_masks.pt" \
  --original_safety_scores_path "${NEURON_DIR}/safety_neurons/original_safety_scores.pt" \
  --fine_tuned_safety_scores_path "${EXP_DIR}/safety_neurons/fine_tuned_safety_scores.pt" \
  --results_path "${EXP_DIR}"

echo "Experiment 1 outputs → ${EXP_DIR}"
