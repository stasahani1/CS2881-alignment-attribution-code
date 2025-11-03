# Guide: Loading Pre-computed Wanda Scores

## Overview

This modification allows `wanda` and `wandg` pruning methods to load pre-computed scores from disk instead of computing them in real-time. This makes the pruning behavior more consistent across methods and helps isolate issues between score computation and pruning logic.

## Changes Made

### 1. New Command-Line Flag
- **`--load_wanda_score`**: Load pre-computed wanda scores instead of computing them
  - Mutually exclusive with `--dump_wanda_score`
  - Works with `wanda`, `wandg`, and `wanda_v2` methods

### 2. Modified Functions
- **`lib/prune.py`**:
  - Added `load_wanda_score()` helper function
  - Modified `prune_wanda()` to skip activation computation when loading scores

- **`lib/model_wrapper.py`**:
  - Modified `prune_wandg()` to skip gradient computation when loading scores
  - Modified `_prune_core()` to support loading scores
  - Modified `prune_wanda_v2()` to pass prune_data parameter

## Usage Workflow

### Step 1: Generate and Save Scores (First Run)
```bash
# For wanda method
python main.py \
    --model llama2-7b-chat-hf \
    --prune_method wanda \
    --prune_data align \
    --sparsity_ratio 0.01 \
    --sparsity_type unstructured \
    --save results/wanda_scores \
    --dump_wanda_score

# For wandg method
python main.py \
    --model llama2-7b-chat-hf \
    --prune_method wandg \
    --prune_data align \
    --sparsity_ratio 0.01 \
    --sparsity_type unstructured \
    --save results/wandg_scores \
    --dump_wanda_score \
    --entangle_prompt_feat  # Required for wandg
```

This will create pickle files in:
- `results/wanda_scores/wanda_score/{prune_data}_weight_only/`
- `results/wandg_scores/wanda_score/`

### Step 2: Load Scores and Prune (Subsequent Runs)
```bash
# For wanda method
python main.py \
    --model llama2-7b-chat-hf \
    --prune_method wanda \
    --prune_data align \
    --sparsity_ratio 0.01 \
    --sparsity_type unstructured \
    --save results/wanda_scores \
    --load_wanda_score \
    --neg_prune \
    --eval_attack \
    --save_attack_res

# For wandg method
python main.py \
    --model llama2-7b-chat-hf \
    --prune_method wandg \
    --prune_data align \
    --sparsity_ratio 0.01 \
    --sparsity_type unstructured \
    --save results/wandg_scores \
    --load_wanda_score \
    --neg_prune \
    --eval_attack \
    --save_attack_res \
    --entangle_prompt_feat
```

## Key Differences vs wandg_set_difference

When using `--load_wanda_score`, the pruning behavior differs from `wandg_set_difference`:

### Pruning Strategy
- **`wanda`/`wandg` with `--load_wanda_score`**:
  - Per-row pruning: sorts along dim=-1 (columns)
  - Each row independently prunes `sparsity_ratio * num_columns` weights
  - Uniform sparsity per row

- **`wandg_set_difference`**:
  - Global pruning: flattens entire weight matrix
  - Selects weights globally across all rows and columns
  - Non-uniform sparsity per row

### Selection Direction
- **`wanda`/`wandg` with `--load_wanda_score`**:
  - With `--neg_prune`: negates scores, then prunes lowest (= highest original)
  - Without `--neg_prune`: prunes lowest scores directly

- **`wandg_set_difference`**:
  - Always selects highest scores via `largest=True`
  - Ignores `--neg_prune` flag

## Debugging Pruning Issues

To test if the issue is in score computation vs pruning logic:

```bash
# 1. Compute and save scores with wanda
python main.py --model llama2-7b-chat-hf --prune_method wanda \
    --prune_data align --sparsity_ratio 0.01 \
    --save results/debug --dump_wanda_score

# 2. Load those same scores and prune
python main.py --model llama2-7b-chat-hf --prune_method wanda \
    --prune_data align --sparsity_ratio 0.01 \
    --save results/debug --load_wanda_score --neg_prune

# 3. Compare with wandg using the same scores
python main.py --model llama2-7b-chat-hf --prune_method wandg \
    --prune_data align --sparsity_ratio 0.01 \
    --save results/debug --load_wanda_score --neg_prune \
    --entangle_prompt_feat
```

If the issue persists with `--load_wanda_score`, the problem is in the pruning logic. If it goes away, the issue is in score computation.

## File Path Formats

The loader tries multiple path formats to find scores:

1. **Primary format** (wanda with disentangle):
   ```
   {save}/wanda_score/{prune_data}_weight_only_disentangle/
       W_metric_layer_{i}_name_{layer_name}_{prune_data}_weight_only_disentangle.pkl
   ```

2. **Alternative format** (wandg):
   ```
   {save}/wanda_score/
       W_metric_layer_{i}_name_{layer_name}_weight.pkl
   ```

## Error Handling

If scores are not found, you'll see:
```
FileNotFoundError: Pre-computed wanda score not found at {path}.
Please run with --dump_wanda_score first to generate scores.
```

Make sure the `--save` directory matches between dump and load runs.

## Performance Benefits

- **Faster iteration**: Skip expensive activation/gradient computation
- **Consistent behavior**: Same scores used across experiments
- **Better debugging**: Isolate score computation from pruning logic
- **Memory efficient**: Don't need to load calibration data during pruning
