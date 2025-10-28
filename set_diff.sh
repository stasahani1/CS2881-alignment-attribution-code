#!/bin/bash

# Enable GPU access - override RunPod's void setting
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

MODEL="llama2-7b-chat-hf"
METHOD="wandg_set_difference"
PRUNE_DATA="align"
OUTPUT_DIR="results/fig2a/snip_setdiff"

# For set difference, we use --p and --q parameters
# Paper mentions top-3% utility and top-3% safety
# Sparsity levels for Figure 2(a): 0% to 3%

SPARSITIES=(0.005 0.010 0.015 0.020 0.025 0.030)

echo "=========================================="
echo "STEP 1: Dumping SNIP scores (this only needs to be done once)"
echo "=========================================="

# Dump SNIP scores for utility dataset (alpaca_cleaned_no_safety)
if [ ! -d "out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score" ]; then
    echo "Dumping SNIP scores for utility dataset (alpaca_cleaned_no_safety)..."
    python main.py \
        --model $MODEL \
        --prune_method wandg \
        --prune_data alpaca_cleaned_no_safety \
        --dump_wanda_score
    echo "Utility scores dumped!"
else
    echo "Utility scores already exist, skipping..."
fi

# Dump SNIP scores for safety dataset (align)
if [ ! -d "out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score" ]; then
    echo "Dumping SNIP scores for safety dataset (align)..."
    python main.py \
        --model $MODEL \
        --prune_method wandg \
        --prune_data $PRUNE_DATA \
        --dump_wanda_score
    echo "Safety scores dumped!"
else
    echo "Safety scores already exist, skipping..."
fi

echo "=========================================="
echo "STEP 2: Running set difference experiments"
echo "=========================================="

for SPARSITY in "${SPARSITIES[@]}"; do
      echo "Running SNIP set difference with p=q=${SPARSITY}"
      SAVE_DIR="${OUTPUT_DIR}/sparsity_${SPARSITY}"

      python main.py \
          --model $MODEL \
          --prune_method $METHOD \
          --prune_data $PRUNE_DATA \
          --sparsity_ratio $SPARSITY \
          --p $SPARSITY \
          --q $SPARSITY \
          --save $SAVE_DIR \
          --eval_zero_shot \
          --eval_attack \
          --save_attack_res

      # Clear GPU memory between iterations
      python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

      echo "Completed p=q=${SPARSITY}"
  done

