#!/bin/bash
################################################################################
# Script: config_example.sh
# Description: Example configuration file for extension experiments
#              Copy this file and modify settings as needed
################################################################################

# ============================================================================
# Model Configuration
# ============================================================================
export MODEL="llama2-7b-chat-hf"  # Options: llama2-7b-chat-hf, llama2-13b-chat-hf

# ============================================================================
# Neuron Identification Settings
# ============================================================================
export PRUNE_METHOD="wanda"       # Options: wanda, wandg
export SPARSITY_RATIO="0.05"      # Fraction of neurons to identify as critical (0.01-0.1)
export NSAMPLES="128"             # Number of samples for scoring (32-256)

# ============================================================================
# Fine-tuning Settings
# ============================================================================
export TRAINING_DATA="alpaca_cleaned_no_safety"  # Dataset for fine-tuning
export NUM_EPOCHS="3"             # Number of training epochs (1-5)
export LEARNING_RATE="2e-5"       # Learning rate (1e-5 to 5e-5)
export BATCH_SIZE="4"             # Batch size (1-8, depends on GPU memory)
export MAX_LENGTH="512"           # Maximum sequence length (256-2048)

# ============================================================================
# Evaluation Settings
# ============================================================================
export EVAL_ATTACK="false"        # Set to "true" to run ASR evaluation (requires vLLM)

# ============================================================================
# System Settings
# ============================================================================
export SEED="42"                  # Random seed for reproducibility
export DEVICE="cuda:0"            # Device to use (cuda:0, cuda:1, etc.)

# ============================================================================
# Usage
# ============================================================================
# Source this file before running scripts:
#   source scripts/config_example.sh
#   ./scripts/run_full_pipeline.sh
#
# Or set variables inline:
#   MODEL=llama2-13b-chat-hf ./scripts/run_full_pipeline.sh
