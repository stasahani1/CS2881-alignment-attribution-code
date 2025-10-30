#!/bin/bash

# Orthogonal Projection Rank Pruning Script
# Identifies safety-critical ranks via orthogonal projection between
# utility projection matrix (Πu) and safety projection matrix (Πs)
# using ActSVD where ranks < 100

MODEL="llama2-7b-chat-hf"
METHOD="low_rank_diff"
TYPE="unstructured"
PRUNE_DATA_POS="alpaca_cleaned_no_safety"  # utility dataset
PRUNE_DATA_NEG="align"                      # safety dataset
OUTPUT_DIR="results/fig2b/orth_proj"

# Rank combinations to test where min(ru, 4096-rs) < 100
# This ensures the orthogonal projection dimension is < 100
# Testing different combinations to identify safety-critical ranks
RANK_COMBINATIONS=(
    "10 4000"      # min(10, 96) = 10
    "20 4000"      # min(20, 96) = 20
    "50 4000"      # min(50, 96) = 50
    "80 4000"      # min(80, 96) = 80
    "90 4000"      # min(90, 96) = 90
    "2550 4000"    # min(2550, 96) = 96
    "3450 4000"    # min(3450, 96) = 96
    "3000 4050"    # min(3000, 46) = 46
    "3500 4080"    # min(3500, 16) = 16
    "4000 4090"    # min(4000, 6) = 6
    "3800 4070"    # min(3800, 26) = 26
    "3200 4040"    # min(3200, 56) = 56
)

echo "=========================================="
echo "Orthogonal Projection Rank Pruning"
echo "Testing rank combinations where min(ru, 4096-rs) < 100"
echo "This identifies safety-critical ranks via orthogonal projection"
echo "=========================================="

for RANKS in "${RANK_COMBINATIONS[@]}"; do
    # Split the rank combination
    RU=$(echo $RANKS | cut -d' ' -f1)
    RS=$(echo $RANKS | cut -d' ' -f2)

    echo "Running orthogonal projection with ru=${RU}, rs=${RS}"
    SAVE_DIR="${OUTPUT_DIR}/ru_${RU}_rs_${RS}"

    python main_low_rank_diff.py \
        --model $MODEL \
        --rank_pos $RU \
        --rank_neg $RS \
        --prune_data_pos $PRUNE_DATA_POS \
        --prune_data_neg $PRUNE_DATA_NEG \
        --save $SAVE_DIR \
        --eval_zero_shot \
        --eval_attack

    # Clear GPU memory between iterations
    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

    # Clean up temporary vLLM model files to prevent disk quota issues
    rm -rf temp/_vllm_tmp temp/tmp_vllm_model

    echo "Completed ru=${RU}, rs=${RS}"
    echo "------------------------------------------"
done

echo "=========================================="
echo "All orthogonal projection experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
