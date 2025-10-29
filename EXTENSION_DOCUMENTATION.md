# Safety-Critical Neuron Analysis and Fine-tuning Extension

## Overview

This extension implements experiments to understand **why safety alignment is brittle** in language models. Building on "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications," we investigate two complementary hypotheses about safety degradation during fine-tuning:

1. **Frozen-Regime Fine-Tuning (Wanda Score Dynamics)**: How does the importance landscape shift when safety-critical neurons are frozen?
2. **Unfrozen Fine-Tuning (Safety Neuron Drift)**: How much do safety-critical neurons change compared to other neurons during normal fine-tuning?

## Research Questions

### Experiment 1: Frozen-Regime Fine-Tuning (Wanda Score Dynamics)

**Setup**: Freeze identified safety-critical neurons and fine-tune on harmless/utility datasets (e.g., Alpaca, harmless instructions). Then re-compute Wanda scores post-fine-tuning.

**Goal**: Quantify whether the relative importance landscape shifts even though those neurons were frozen.

**Hypotheses**:
- **Hypothesis A (Representational Drift)**: If Wanda scores of frozen safety neurons drop sharply, it suggests representational drift elsewhere causes safety-relevant pathways to degrade. The frozen neurons become "stranded" as the model reorganizes around them.
- **Hypothesis B (Stability Under Freezing)**: If their Wanda scores stay stable, brittleness might stem from global redistribution of function rather than direct parameter change. Safety persists locally but fails globally.

**Key Metrics**:
- Wanda score distributions before/after (mean, std, top-k percentiles)
- Score drop percentage for frozen safety-critical neurons
- Attack Success Rate (ASR) before/after fine-tuning

### Experiment 2: Unfrozen Fine-Tuning (Safety Neuron Drift)

**Setup**: Fine-tune normally (no freezing) and measure how much each safety-critical neuron changes.

**Goal**: Determine if safety-critical neurons are particularly fragile (high drift) or if safety degradation comes from new harmful circuits activating elsewhere.

**Hypotheses**:
- **Hypothesis C (Fragile Safety Neurons)**: Safety-critical neurons move more than average → safety is inherently "fragile" and easily disrupted.
- **Hypothesis D (Recontextualization)**: They move less but are recontextualized—fine-tuning activates new harmful circuits instead of destroying old safety ones.

**Key Metrics**:
- Cosine similarity between pre/post fine-tune weight vectors per neuron
- L2 distance of weight changes
- Comparison: safety-critical vs. random vs. utility-critical neurons
- Attack Success Rate (ASR) before/after fine-tuning

## Architecture

The extension is organized into two main modules:

- **`main_extension.py`**: Main entry point with CLI interface and pipeline orchestration
- **`lib/extension_utils.py`**: Core classes and utilities for safety neuron analysis and fine-tuning

## Key Features

### 1. Safety-Critical Neuron Identification
- **Wanda Scoring**: Magnitude-based scoring (`|w| × √(activation_norm)`) - primary method
- **SNIP (WandG) Scoring**: Gradient-based importance scoring (`|∇w|`) - alternative method
- **Safety Dataset Support**: Works with aligned datasets (`align`, `align_short`) to identify safety-relevant neurons
- **Utility-Critical Neuron Identification**: Also identifies utility-critical neurons using `alpaca_cleaned_no_safety`
- **Configurable Sparsity**: Adjustable fraction of neurons to identify as critical (typically 0.01-0.1)

### 2. Frozen-Regime Fine-Tuning
- **Gradient Hooks**: Implements PyTorch hooks to zero out gradients for safety-critical neurons
- **Selective Freezing**: Only freezes identified safety-critical neurons while allowing others to update
- **Memory Efficient**: Preserves original model structure while preventing updates to critical neurons
- **Wanda Score Tracking**: Re-computes Wanda scores post-fine-tuning to measure importance landscape shifts

### 3. Unfrozen Fine-Tuning with Drift Measurement
- **Weight Snapshot**: Captures pre-fine-tuning weights for all neurons
- **Drift Metrics**: Computes cosine similarity and L2 distance for each neuron
- **Comparative Analysis**: Measures drift for safety-critical vs. random vs. utility-critical neurons
- **Statistical Testing**: Determines if safety neurons drift significantly more/less than baseline

### 4. Evaluation and Analysis
- **Attack Success Rate (ASR)**: Uses `eval_attack()` from base codebase to measure safety degradation
- **Score Dynamics**: Tracks Wanda score changes for frozen safety neurons
- **Weight Drift Analysis**: Quantifies parameter changes per neuron category
- **Visualization Ready**: Exports data for plotting score distributions and drift comparisons

## Implementation Details

### Core Classes (`lib/extension_utils.py`)

#### `SafetyNeuronAnalyzer`
```python
class SafetyNeuronAnalyzer:
    def identify_safety_critical_neurons(self, prune_method, prune_data, sparsity_ratio, nsamples, seed)
```

**Key Methods:**
- `identify_safety_critical_neurons()`: Uses existing pruning infrastructure to identify safety-critical neurons using score-only approach (no actual pruning)
- Returns both safety masks and importance scores for further analysis
- Can identify utility-critical neurons by passing `prune_data="alpaca_cleaned_no_safety"`

#### `FineTuner`
```python
class FineTuner:
    def __init__(self, model, tokenizer, device, safety_masks, original_weights)
    def fine_tune_model(self, training_data, num_epochs, learning_rate, batch_size, max_length,
                       model_save_path, freeze_neurons)
    def _freeze_safety_critical_neurons(self)
    def compute_weight_drift(self, neuron_masks_dict)
```

**Key Methods:**
- `fine_tune_model()`: Implements the fine-tuning loop using HuggingFace Trainer
  - `freeze_neurons=True`: Frozen-regime experiment (Hypothesis A/B)
  - `freeze_neurons=False`: Unfrozen experiment (Hypothesis C/D)
- `_freeze_safety_critical_neurons()`: Implements gradient hooks to prevent updates to critical neurons
- `compute_weight_drift()`: Measures cosine similarity and L2 distance for safety/utility/random neurons

#### `ScoreDynamicsAnalyzer`
```python
class ScoreDynamicsAnalyzer:
    def compare_score_distributions(self, original_scores, new_scores, safety_masks)
    def compute_score_drop_percentage(self, original_scores, new_scores, safety_masks)
```

**Key Methods:**
- `compare_score_distributions()`: Statistical comparison of Wanda scores before/after fine-tuning
- `compute_score_drop_percentage()`: Measures how much frozen safety neurons' scores dropped

### Integration with Existing Codebase

The extension **reuses as much code as possible** from the alignment attribution codebase:

1. **Model Loading**: Uses `get_llm()` and `modeltype2path` from `main.py` for consistent model loading
2. **Data Loading**: Utilizes `get_loaders()` from `lib.data` for dataset handling (align, alpaca_cleaned_no_safety)
3. **Model Wrappers**: Integrates with `make_Act`, `revert_Act_to_Linear`, and `ActLinear` from `lib.model_wrapper`
4. **Pruning Utilities**: Uses `find_layers` and `check_sparsity` from `lib.prune`
5. **Evaluation**: **Directly reuses `eval_attack()` from `lib.eval`** to measure ASR on jailbreak datasets
6. **Score Computation**: Reuses Wanda/SNIP scoring logic from existing pruning methods

## Usage

### Quick Start (Recommended)

**Using automated scripts** (easiest method):

```bash
# Quick test to verify everything works (~30-60 min)
./scripts/quick_test.sh

# Full pipeline with default settings
./scripts/run_full_pipeline.sh

# Custom configuration
source scripts/config_example.sh
./scripts/run_full_pipeline.sh
```

See [scripts/README.md](scripts/README.md) for detailed script documentation.

### Manual Execution

If you prefer to run commands manually or need fine-grained control:

### Experiment 1: Frozen-Regime Fine-Tuning (Score Dynamics)

**Step 1: Identify safety-critical neurons**
```bash
uv run python main_extension.py \
    --task identify_safety_neurons \
    --model llama2-7b-chat-hf \
    --prune_method wanda \
    --prune_data align_short \
    --sparsity_ratio 0.05 \
    --nsamples 128 \
    --results_path ./results/frozen_regime
```

**Step 2: Fine-tune with frozen neurons**
```bash
uv run python main_extension.py \
    --task fine_tune_frozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./results/frozen_regime/safety_neurons/original_safety_masks.pt \
    --training_data alpaca_cleaned_no_safety \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --model_save_path ./fine_tuned_frozen_model \
    --results_path ./results/frozen_regime
```

**Step 3: Evaluate score dynamics and ASR**
```bash
uv run python main_extension.py \
    --task eval_score_dynamics \
    --model llama2-7b-chat-hf \
    --original_model_path ./models/llama-2-7b-chat-hf \
    --fine_tuned_model_path ./fine_tuned_frozen_model \
    --original_safety_masks_path ./results/frozen_regime/safety_neurons/original_safety_masks.pt \
    --original_safety_scores_path ./results/frozen_regime/safety_neurons/original_safety_scores.pt \
    --fine_tuned_safety_scores_path ./results/frozen_regime/safety_neurons/fine_tuned_safety_scores.pt \
    --results_path ./results/frozen_regime \
    --eval_attack
```

### Experiment 2: Unfrozen Fine-Tuning (Weight Drift)

**Step 1: Identify safety-critical and utility-critical neurons**
```bash
# Identify safety-critical neurons
uv run python main_extension.py \
    --task identify_safety_neurons \
    --model llama2-7b-chat-hf \
    --prune_method wanda \
    --prune_data align_short \
    --sparsity_ratio 0.05 \
    --results_path ./results/unfrozen_regime

# Identify utility-critical neurons
uv run python main_extension.py \
    --task identify_utility_neurons \
    --model llama2-7b-chat-hf \
    --prune_method wanda \
    --prune_data alpaca_cleaned_no_safety \
    --sparsity_ratio 0.05 \
    --results_path ./results/unfrozen_regime
```

**Step 2: Fine-tune without freezing (capture weight drift)**
```bash
uv run python main_extension.py \
    --task fine_tune_unfrozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./results/unfrozen_regime/safety_neurons/original_safety_masks.pt \
    --utility_masks ./results/unfrozen_regime/utility_neurons/original_utility_masks.pt \
    --training_data alpaca_cleaned_no_safety \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --model_save_path ./fine_tuned_unfrozen_model \
    --results_path ./results/unfrozen_regime
```

**Step 3: Evaluate weight drift and ASR**
```bash
uv run python main_extension.py \
    --task eval_weight_drift \
    --original_model_path ./models/llama-2-7b-chat-hf \
    --fine_tuned_model_path ./fine_tuned_unfrozen_model \
    --safety_masks_path ./results/unfrozen_regime/safety_neurons/original_safety_masks.pt \
    --utility_masks_path ./results/unfrozen_regime/utility_neurons/original_utility_masks.pt \
    --original_weights_path ./results/unfrozen_regime/original_weights.pt \
    --results_path ./results/unfrozen_regime \
    --eval_attack
```

## Command Line Arguments

### Task Arguments
- `--task`: Task to perform
  - `identify_safety_neurons`: Identify safety-critical neurons using Wanda/SNIP scores
  - `identify_utility_neurons`: Identify utility-critical neurons
  - `fine_tune_frozen`: Fine-tune with frozen safety-critical neurons (Experiment 1)
  - `fine_tune_unfrozen`: Fine-tune normally and track weight drift (Experiment 2)
  - `eval_score_dynamics`: Evaluate Wanda score changes for Experiment 1
  - `eval_weight_drift`: Evaluate neuron weight drift for Experiment 2

### Model Arguments
- `--model`: Model name to analyze (default: `llama2-7b-chat-hf`)
- `--original_model_path`: Path to original pre-fine-tuned model (for evaluation tasks)
- `--fine_tuned_model_path`: Path to fine-tuned model (for evaluation tasks)

### Neuron Identification Arguments
- `--prune_method`: Method for identifying critical neurons (`wanda` or `wandg`, default: `wanda`)
- `--prune_data`: Dataset for neuron analysis
  - `align` or `align_short`: for safety-critical neurons
  - `alpaca_cleaned_no_safety`: for utility-critical neurons
- `--sparsity_ratio`: Fraction of neurons to identify as critical (default: `0.05`)
- `--nsamples`: Number of samples for scoring (default: `128`)

### Fine-tuning Arguments
- `--safety_masks`: Path to safety neuron masks (.pt file)
- `--utility_masks`: Path to utility neuron masks (.pt file, for Experiment 2)
- `--training_data`: Dataset for fine-tuning (default: `alpaca_cleaned_no_safety`)
- `--num_epochs`: Number of training epochs (default: `3`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--batch_size`: Batch size (default: `4`)
- `--max_length`: Maximum sequence length (default: `512`)

### Evaluation Arguments
- `--original_safety_masks_path`: Path to original safety neuron masks
- `--original_safety_scores_path`: Path to original safety neuron scores
- `--fine_tuned_safety_scores_path`: Path to fine-tuned safety neuron scores
- `--safety_masks_path`: Path to safety masks for drift analysis
- `--utility_masks_path`: Path to utility masks for drift analysis
- `--original_weights_path`: Path to pre-fine-tuning weights snapshot
- `--eval_attack`: Flag to run ASR evaluation using `eval_attack()` from base codebase

### Output Arguments
- `--model_save_path`: Path to save fine-tuned model (default: `./fine_tuned_model`)
- `--results_path`: Path to save analysis results (default: `./results`)

### Other Arguments
- `--seed`: Random seed (default: `42`)
- `--device`: Device to use (default: `cuda:0`)

## File Transfer between Local and Remote 

Transfer `results/` directory from local to remote by: 
```bash 
scp -r results/ persona-vector:/workspace/CS2881-alignment-attribution-code
```

Transfer results directory from remote to local by: 
```bash 
scp -r persona-vector:/workspace/CS2881-alignment-attribution-code/results .
```