# Summary of Changes: Load Pre-computed Wanda Scores

## Problem Identified

The original code had discrepancies between how `wanda`/`wandg` and `wandg_set_difference` perform pruning:

1. **Pruning Strategy**:
   - `wanda`/`wandg`: Per-row pruning (sorts along dim=-1)
   - `wandg_set_difference`: Global pruning (flattens entire matrix)

2. **Selection Direction**:
   - `wanda`/`wandg`: Prunes lowest scores (with `--neg_prune` to prune highest)
   - `wandg_set_difference`: Always selects highest scores, ignores `--neg_prune`

3. **Score Computation**:
   - `wanda`/`wandg`: Computes scores in real-time
   - `wandg_set_difference`: Loads pre-computed scores from disk

## Solution: Unified Score Loading

Added the ability to load pre-computed scores for `wanda` and `wandg`, making the workflow consistent across all methods and allowing better debugging.

## Files Modified

### 1. `main.py`
**Added:**
- New argument `--load_wanda_score` (line 159-161)
- Validation to prevent simultaneous use of `--dump_wanda_score` and `--load_wanda_score` (lines 196-197)

### 2. `lib/prune.py`
**Added:**
- `load_wanda_score()` function (lines 151-206) - Helper to load pre-computed scores with multiple path format support

**Modified:**
- `prune_wanda()` function:
  - Skip calibration data loading when `--load_wanda_score` is set (lines 327-362)
  - Skip activation computation when loading scores (lines 381-419)
  - Load scores instead of computing them in main pruning loop (lines 426-453)
  - Load scores in prune_part section (lines 591-618)
  - Skip layer output computation when loading scores (lines 735-743)

### 3. `lib/model_wrapper.py`
**Added:**
- Import of `load_wanda_score` function (line 13)

**Modified:**
- `prune_wandg()` function:
  - Skip gradient computation when `--load_wanda_score` is set (lines 254-318)
  - Added branch to load scores without computing gradients (lines 319-334)
  - Pass `prune_data` parameter to `_prune_core()` (line 316)

- `_prune_core()` function:
  - Added `prune_data` parameter (line 348)
  - Load scores instead of computing when `--load_wanda_score` is set (lines 376-407)

- `prune_wanda_v2()` function:
  - Pass `prune_data` parameter to `_prune_core()` (line 187)

- `prune_wandg_v1()` function:
  - Pass `prune_data` parameter to `_prune_core()` (line 236)

## New Files Created

### 1. `LOAD_SCORES_GUIDE.md`
Comprehensive usage guide including:
- Overview of changes
- Step-by-step workflow
- Key differences from `wandg_set_difference`
- Debugging instructions
- File path formats
- Error handling

### 2. `test_load_scores.sh`
Example shell script demonstrating:
- Dumping scores with `wanda`
- Loading scores with `wanda`
- Dumping scores with `wandg`
- Loading scores with `wandg`
- Comparison with `wandg_set_difference`

### 3. `CHANGES_SUMMARY.md` (this file)
Summary of all modifications

## Usage Example

```bash
# 1. Generate scores
python main.py --model llama2-7b-chat-hf --prune_method wanda \
    --prune_data align --sparsity_ratio 0.01 \
    --save results/scores --dump_wanda_score

# 2. Load scores and prune
python main.py --model llama2-7b-chat-hf --prune_method wanda \
    --prune_data align --sparsity_ratio 0.01 \
    --save results/scores --load_wanda_score --neg_prune \
    --eval_attack --save_attack_res
```

## Benefits

1. **Consistency**: Same workflow for all pruning methods
2. **Performance**: Skip expensive activation/gradient computation
3. **Debugging**: Isolate score computation from pruning logic
4. **Reproducibility**: Reuse same scores across experiments
5. **Memory Efficiency**: Don't need calibration data during pruning

## Debugging the Original Issue

To test whether the issue is in score computation vs pruning logic:

1. Generate scores with `wandg`: `--dump_wanda_score`
2. Load those scores with `wanda`: `--load_wanda_score`
3. Compare results

If results differ, the issue is in the pruning logic (per-row vs global). If they're the same, the issue is in score computation.

## Backward Compatibility

All changes are backward compatible:
- Existing scripts without `--load_wanda_score` work unchanged
- Default behavior remains computing scores in real-time
- Only adds new functionality, doesn't remove or break existing features
