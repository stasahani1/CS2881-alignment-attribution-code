# Safety-Critical Neuron Analysis and Fine-tuning Extension

## Overview

This extension implements a comprehensive pipeline for analyzing safety-critical neurons in language models and studying how they change during fine-tuning. The implementation follows the research framework from "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications" and extends it with a novel approach to freeze safety-critical neurons during fine-tuning.

## Architecture

The extension is organized into two main modules:

- **`main_extension.py`**: Main entry point with CLI interface and pipeline orchestration
- **`lib/extension_utils.py`**: Core classes and utilities for safety neuron analysis and fine-tuning

## Key Features

### 1. Safety-Critical Neuron Identification
- **SNIP (WandG) Scoring**: Uses gradient-based importance scoring to identify neurons critical for safety
- **Wanda Scoring**: Alternative magnitude-based scoring method
- **Safety Dataset Support**: Works with aligned datasets (`align`, `align_short`) to identify safety-relevant neurons
- **Configurable Sparsity**: Adjustable fraction of neurons to identify as safety-critical

### 2. Neuron Freezing Mechanism
- **Gradient Hooks**: Implements PyTorch hooks to zero out gradients for safety-critical neurons
- **Selective Freezing**: Only freezes identified safety-critical neurons while allowing others to update
- **Memory Efficient**: Preserves original model structure while preventing updates to critical neurons

### 3. Fine-tuning with Frozen Neurons
- **HuggingFace Integration**: Uses Transformers library for robust fine-tuning
- **Multiple Dataset Support**: Compatible with various training datasets
- **Configurable Training**: Adjustable epochs, learning rate, batch size, and sequence length
- **Model Persistence**: Saves fine-tuned models for further analysis

### 4. Post-Fine-tuning Analysis
- **Score Recalculation**: Recomputes SNIP/Wanda scores on the fine-tuned model
- **Change Analysis**: Compares safety-critical neurons before and after fine-tuning
- **Quantitative Metrics**: Provides overlap statistics and change analysis

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

#### `FineTuner`
```python
class FineTuner:
    def __init__(self, model, tokenizer, device, safety_masks)
    def fine_tune_model(self, training_data, num_epochs, learning_rate, batch_size, max_length, model_save_path)
    def _freeze_safety_critical_neurons(self)
```

**Key Methods:**
- `fine_tune_model()`: Implements the fine-tuning loop using HuggingFace Trainer
- `_freeze_safety_critical_neurons()`: Implements gradient hooks to prevent updates to critical neurons
- Automatically applies neuron freezing before training begins

### Integration with Existing Codebase

The extension leverages existing infrastructure from the alignment attribution codebase:

1. **Model Loading**: Uses `get_llm()` and `modeltype2path` from `main.py` for consistent model loading
2. **Data Loading**: Utilizes `get_loaders()` from `lib.data` for dataset handling
3. **Model Wrappers**: Integrates with `make_Act`, `revert_Act_to_Linear`, and `ActLinear` from `lib.model_wrapper`
4. **Pruning Utilities**: Uses `find_layers` and `check_sparsity` from `lib.prune`
5. **Evaluation**: Compatible with existing evaluation functions in `lib.eval`

## Usage

### Basic Usage

```bash
uv run python main_extension.py \
    --task identify_safety_neurons \
    --model llama2-7b-chat-hf \
    --prune_method wandg \
    --prune_data align_short \
    --sparsity_ratio 0.05 \
    --results_path ./results
```

### Advanced Configuration

```bash
uv run python main_extension.py \
    --task identify_safety_neurons \
    --model llama2-7b-chat-hf \
    --prune_method wanda \
    --prune_data align_short \
    --sparsity_ratio 0.1 \
    --nsamples 256 \
    --results_path ./results/custom_results \
    --seed 123
```

## Command Line Arguments

### Task Arguments
- `--task`: Task to perform (`identify_safety_neurons`, `fine_tune`, or `eval`)

### Model Arguments
- `--model`: Model name to analyze (default: `llama2-7b-chat-hf`)

### Safety Analysis Arguments
- `--prune_method`: Method for identifying safety-critical neurons (`wandg` or `wanda`, default: `wandg`)
- `--prune_data`: Dataset for safety analysis (`align` or `align_short`, default: `align`)
- `--sparsity_ratio`: Fraction of neurons to identify as safety-critical (default: `0.1`)
- `--nsamples`: Number of samples for safety analysis (default: `128`)

### Fine-tuning Arguments
- `--safety_masks`: Path to safety data file for freezing neurons
- `--training_data`: Dataset for fine-tuning (default: `alpaca_cleaned_no_safety`)
- `--num_epochs`: Number of training epochs (default: `3`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--batch_size`: Batch size (default: `4`)
- `--max_length`: Maximum sequence length (default: `512`)

### Evaluation Arguments
- `--original_safety_masks_path`: Path to original safety data file
- `--fine_tuned_safety_masks_path`: Path to fine-tuned safety data file

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