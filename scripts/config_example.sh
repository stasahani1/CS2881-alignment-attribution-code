#!/bin/bash
################################################################################
# Example configuration for extension experiments
# Copy and modify as needed, then: source scripts/config_example.sh
################################################################################

# Model Configuration
export MODEL="llama2-7b-chat-hf"

# Neuron Identification Settings
export PRUNE_METHOD="wanda"
export SPARSITY_RATIO="0.05"
export NSAMPLES="128"

# Fine-tuning Settings
export TRAINING_DATA="alpaca_cleaned_no_safety"
export NUM_EPOCHS="3"
export LEARNING_RATE="2e-5"
export BATCH_SIZE="4"
export MAX_LENGTH="512"

# PEFT/LoRA Settings (optional)
export USE_PEFT="false"           # Set to "true" to enable PEFT
export PEFT_R="8"                 # LoRA rank (default: 8)
export PEFT_ALPHA="16"            # LoRA alpha (default: 16)
export PEFT_TARGET_MODULES="q_proj,v_proj"  # Target modules

# Evaluation & System Settings
export EVAL_ATTACK="false"
export SEED="42"
export DEVICE="cuda:0"

# Usage:
#   source scripts/config_example.sh && ./scripts/run_full_pipeline.sh
#   USE_PEFT=true PEFT_R=16 ./scripts/01_run_experiment_1_frozen.sh
