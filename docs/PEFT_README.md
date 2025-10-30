# PEFT (LoRA) Fine-Tuning Extension

## Overview

This extension adds PEFT (Parameter-Efficient Fine-Tuning) support via LoRA to the fine-tuning experiments. It enables testing whether LoRA adapters can create "alternative pathways" that bypass frozen safety-critical neurons.

## What Was Added

### 1. PEFT Support in `FineTuner` Class ([lib/extension_utils.py](lib/extension_utils.py))

**New Parameters:**
- `use_peft`: Enable PEFT/LoRA fine-tuning
- `peft_r`: LoRA rank (default: 8)
- `peft_alpha`: LoRA alpha scaling (default: 16)
- `peft_target_modules`: Target modules for LoRA (default: "q_proj,v_proj")

**Key Features:**
- Wraps model with LoRA adapters when `use_peft=True`
- Freezing mechanism works on base model neurons (LoRA adapters train freely)
- Tests "alternative pathways" hypothesis: Can LoRA bypass frozen safety neurons?

### 2. PEFT Model Handling in `SafetyNeuronAnalyzer` ([lib/extension_utils.py](lib/extension_utils.py))

**New Method:**
- `_unwrap_peft_model()`: Unwraps PEFT models to access base model for Wanda scoring

**Behavior:**
- Automatically detects PEFT-wrapped models
- Computes Wanda scores on base model (not adapters)
- Ensures consistent scoring between PEFT and non-PEFT models

### 3. CLI Parameters ([main_extension.py](main_extension.py))

**New Arguments:**
```bash
--use_peft                     # Enable PEFT/LoRA fine-tuning
--peft_r 8                     # LoRA rank (default: 8)
--peft_alpha 16                # LoRA alpha (default: 16)
--peft_target_modules q_proj,v_proj  # Target modules (default: q_proj,v_proj)
```

### 4. Plotting Utility ([plot_wanda_scores.py](plot_wanda_scores.py))

**Features:**
- Visualizes Wanda score distributions before/after fine-tuning
- Plots score changes for frozen safety-critical neurons
- Compares statistical metrics (mean, median, std)
- Provides interpretation hints (Hypothesis A vs B)

**Usage:**
```bash
python plot_wanda_scores.py \
    --original_scores ./results/safety_neurons/original_safety_scores.pt \
    --fine_tuned_scores ./results/safety_neurons/fine_tuned_safety_scores.pt \
    --masks ./results/safety_neurons/original_safety_masks.pt \
    --output_dir ./plots
```

## How LoRA Works with Frozen Neurons

### Architecture

**Without PEFT:**
```
y = Wx
```
Where `W` are the base model weights.

**With PEFT (LoRA):**
```
y = Wx + BAx
```
Where:
- `W`: Base model weights (frozen)
- `B`, `A`: LoRA adapter matrices (trainable)

### Freezing Behavior

When safety-critical neurons are frozen with PEFT enabled:

1. **Base model neurons (W)** are frozen via gradient hooks
2. **LoRA adapters (B, A)** remain fully trainable
3. LoRA adapters can create "alternative pathways" around frozen neurons
4. Output is modified via residual connection: `W + BA`

### Testing Hypotheses

This setup directly tests the paper's "alternative pathways" theory:

- **Hypothesis A (Representational Drift)**: Frozen neurons' Wanda scores drop as LoRA adapters redistribute computation
- **Hypothesis B (Global Redistribution)**: Scores remain stable; safety degradation comes from global reorganization, not local drift

## Usage Examples

### Basic PEFT Fine-Tuning

```bash
# Step 1: Identify safety-critical neurons (same as before)
python main_extension.py \
    --task identify_safety_neurons \
    --model llama2-7b-chat-hf \
    --sparsity_ratio 0.03 \
    --results_path ./results/peft_experiment

# Step 2: Fine-tune with PEFT + frozen neurons
python main_extension.py \
    --task fine_tune_frozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./results/peft_experiment/safety_neurons/original_safety_masks.pt \
    --use_peft \
    --peft_r 8 \
    --peft_alpha 16 \
    --results_path ./results/peft_experiment

# Step 3: Evaluate Wanda score changes
python main_extension.py \
    --task eval_score_dynamics \
    --original_safety_scores_path ./results/peft_experiment/safety_neurons/original_safety_scores.pt \
    --fine_tuned_safety_scores_path ./results/peft_experiment/safety_neurons/fine_tuned_safety_scores.pt \
    --original_safety_masks_path ./results/peft_experiment/safety_neurons/original_safety_masks.pt \
    --results_path ./results/peft_experiment

# Step 4: Plot results
python plot_wanda_scores.py \
    --original_scores ./results/peft_experiment/safety_neurons/original_safety_scores.pt \
    --fine_tuned_scores ./results/peft_experiment/safety_neurons/fine_tuned_safety_scores.pt \
    --masks ./results/peft_experiment/safety_neurons/original_safety_masks.pt \
    --output_dir ./results/peft_experiment/plots
```

### Advanced: Custom LoRA Configuration

```bash
# Use higher rank LoRA with more target modules
python main_extension.py \
    --task fine_tune_frozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./results/safety_neurons/original_safety_masks.pt \
    --use_peft \
    --peft_r 16 \
    --peft_alpha 32 \
    --peft_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --results_path ./results/peft_r16
```

### Compare PEFT vs Full Fine-Tuning

```bash
# Run both experiments
# Full fine-tuning (baseline)
python main_extension.py --task fine_tune_frozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./results/safety_neurons/original_safety_masks.pt \
    --results_path ./results/full_ft

# PEFT fine-tuning
python main_extension.py --task fine_tune_frozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./results/safety_neurons/original_safety_masks.pt \
    --use_peft \
    --results_path ./results/peft_ft

# Compare results
python plot_wanda_scores.py \
    --original_scores ./results/safety_neurons/original_safety_scores.pt \
    --fine_tuned_scores ./results/full_ft/safety_neurons/fine_tuned_safety_scores.pt \
    --masks ./results/safety_neurons/original_safety_masks.pt \
    --output_dir ./results/full_ft/plots

python plot_wanda_scores.py \
    --original_scores ./results/safety_neurons/original_safety_scores.pt \
    --fine_tuned_scores ./results/peft_ft/safety_neurons/fine_tuned_safety_scores.pt \
    --masks ./results/safety_neurons/original_safety_masks.pt \
    --output_dir ./results/peft_ft/plots
```

## Technical Details

### LoRA Configuration

**Default Configuration (Efficient):**
```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                              # Low rank (0.1-0.5% parameters)
    lora_alpha=16,                    # 2x rank (standard)
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Attention only
    bias="none"
)
```

**Why These Defaults?**
- `r=8`: Balances efficiency vs. capacity (matches "low-cost" theme from paper)
- `lora_alpha=16`: Standard 2x rank scaling
- `q_proj, v_proj`: Most efficient adapter placement (attention queries and values)
- Can be adjusted via CLI for experimentation

### Memory & Performance

**Benefits of PEFT:**
- **Memory**: ~30% reduction vs. full fine-tuning (fewer optimizer states)
- **Speed**: ~30% faster training (fewer parameters to update)
- **Storage**: Adapters are ~10-50MB vs. 13GB full model
- **Flexibility**: Multiple adapters can be trained and swapped

**Comparison:**
| Method | Trainable Params | GPU Memory | Training Time | Save Size |
|--------|------------------|------------|---------------|-----------|
| Full FT | 100% (~7B) | ~40GB | 1.0x | 13GB |
| LoRA (r=8) | ~0.3% (~20M) | ~30GB | 0.7x | 40MB |

### Integration Details

**Gradient Hook Mechanism:**
- Hooks registered on base model parameters (not LoRA adapters)
- Base model gradients zeroed for frozen neurons
- LoRA adapter gradients flow normally
- Combined output: `frozen_base + trainable_adapters`

**Wanda Score Computation:**
- Model unwrapped to access base weights
- Scores computed on base model only (adapters ignored)
- Consistent scoring methodology across PEFT/non-PEFT

**Saving & Loading:**
- PEFT models save base + adapters separately
- Use `model.save_pretrained()` to save adapters
- Use `model.base_model.save_pretrained()` to save base model

## Expected Results

### Hypothesis A (Representational Drift)
If Wanda scores of frozen safety neurons **drop significantly** (>10%):
- LoRA adapters successfully redistribute computation around frozen neurons
- Frozen neurons become "stranded" as model reorganizes
- Supports fragility of safety alignment

### Hypothesis B (Global Redistribution)
If Wanda scores **remain stable** (<5% change):
- Frozen neurons maintain importance locally
- Safety degradation from global reorganization, not local drift
- Suggests safety is contextual, not localized

## Files Modified/Created

**Modified:**
1. [lib/extension_utils.py](lib/extension_utils.py)
   - Extended `FineTuner.__init__()` with PEFT parameters
   - Added `_apply_peft()` method
   - Updated `_freeze_safety_critical_neurons()` for PEFT compatibility
   - Added `_get_base_model()` helper
   - Updated `SafetyNeuronAnalyzer.identify_safety_critical_neurons()` to unwrap PEFT
   - Added `_unwrap_peft_model()` method

2. [main_extension.py](main_extension.py)
   - Added `--use_peft`, `--peft_r`, `--peft_alpha`, `--peft_target_modules` arguments
   - Wired up PEFT parameters to `FineTuner` in `fine_tune_frozen` task

**Created:**
3. [plot_wanda_scores.py](plot_wanda_scores.py)
   - New plotting utility for visualizing Wanda score changes
   - Generates histograms, statistical comparisons, and interpretations

4. [PEFT_README.md](PEFT_README.md) (this file)
   - Documentation for PEFT extension

## Testing

### Verify Installation

```bash
# Check PEFT library
python -c "from peft import LoraConfig, get_peft_model; print('✓ PEFT available')"

# Check syntax
python -c "import ast; ast.parse(open('lib/extension_utils.py').read()); print('✓ Syntax OK')"
```

### Minimal Test

```bash
# Small-scale test (fast, verifies everything works)
python main_extension.py \
    --task identify_safety_neurons \
    --model llama2-7b-chat-hf \
    --sparsity_ratio 0.01 \
    --nsamples 32 \
    --results_path ./test_results

python main_extension.py \
    --task fine_tune_frozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./test_results/safety_neurons/original_safety_masks.pt \
    --use_peft \
    --num_epochs 1 \
    --nsamples 100 \
    --results_path ./test_results
```

## Troubleshooting

### Import Error: No module named 'peft'
```bash
pip install peft
```

### Import Error: No module named 'matplotlib'
```bash
pip install matplotlib numpy
```

### PEFT Model Not Wrapping
- Ensure `--use_peft` flag is set
- Check logs for "Applying PEFT (LoRA)" message
- Verify PEFT library version: `pip show peft` (should be 0.6.2+)

### Wanda Scores Not Computing on PEFT Model
- Check logs for "Detected PEFT model, unwrapping to base model for scoring..."
- Ensure `SafetyNeuronAnalyzer._unwrap_peft_model()` is called
- Verify base model is accessible: `model.base_model.model`

## References

- **PEFT Library**: https://github.com/huggingface/peft
- **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **Base Paper**: "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications"
