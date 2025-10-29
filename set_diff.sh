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

# Use /dev/shm to avoid disk quota issues
SCORE_BASE="/dev/shm/wanda_scores"
mkdir -p "$SCORE_BASE"

echo "=========================================="
echo "STEP 0: Setting up symlinks to /dev/shm (to avoid disk quota)"
echo "=========================================="

# Create symlink structure if it doesn't exist
if [ ! -L "out" ]; then
    if [ -d "out" ]; then
        echo "Moving existing out/ to /dev/shm..."
        mv out "$SCORE_BASE/out"
    else
        mkdir -p "$SCORE_BASE/out"
    fi
    ln -s "$SCORE_BASE/out" out
    echo "Created symlink: out -> $SCORE_BASE/out"
else
    echo "Symlink already exists, skipping..."
fi

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
        --sparsity_ratio 0.01 \
        --dump_wanda_score \
        --save out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety
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
        --sparsity_ratio 0.01 \
        --dump_wanda_score \
        --save out/llama2-7b-chat-hf/unstructured/wandg/align
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
