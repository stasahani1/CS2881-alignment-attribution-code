# Safety-Critical Neuron Analysis and Fine-tuning Extension

## Overview

This extension implements a comprehensive pipeline for analyzing safety-critical neurons in language models and studying how they change during fine-tuning. The implementation follows the research framework from "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications" and extends it with a novel approach to freeze safety-critical neurons during fine-tuning.

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

### Core Classes

#### `SafetyNeuronAnalyzer`
```python
class SafetyNeuronAnalyzer:
    def identify_safety_critical_neurons(self, prune_method, prune_data, sparsity_ratio, nsamples, seed)
    def freeze_safety_critical_neurons(self, model)
    def _store_original_weights(self, model)
    def _extract_safety_masks(self, pruned_model)
```

**Key Methods:**
- `identify_safety_critical_neurons()`: Uses existing pruning infrastructure to identify safety-critical neurons
- `freeze_safety_critical_neurons()`: Implements gradient hooks to prevent updates to critical neurons
- `_extract_safety_masks()`: Extracts boolean masks indicating which neurons were pruned

#### `FineTuner`
```python
class FineTuner:
    def fine_tune_model(self, training_data, num_epochs, learning_rate, batch_size, max_length, save_path)
```

**Key Methods:**
- `fine_tune_model()`: Implements the fine-tuning loop using HuggingFace Trainer

### Integration with Existing Codebase

The extension leverages existing infrastructure from the alignment attribution codebase:

1. **Pruning Functions**: Uses `prune_wandg()` and `prune_wanda()` from `lib.prune`
2. **Data Loading**: Utilizes `get_loaders()` from `lib.data` for dataset handling
3. **Model Wrappers**: Integrates with `ActLinear` and activation recording from `lib.model_wrapper`
4. **Evaluation**: Compatible with existing evaluation functions in `lib.eval`

## Usage

### Basic Usage

```bash
uv run python extension.py \
    --model llama2-7b-chat-hf \
    --prune_method wandg \
    --prune_data align_short \
    --sparsity_ratio 0.1 \
    --training_data alpaca_cleaned_no_safety \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --save_path ./models/fine_tuned_model \
    --results_path ./results
```

### Advanced Configuration

```bash
python extension.py \
    --model llama2-7b-chat-hf \
    --prune_method wanda \
    --prune_data align_short \
    --sparsity_ratio 0.2 \
    --nsamples 256 \
    --training_data alpaca_cleaned_no_safety \
    --num_epochs 5 \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --max_length 1024 \
    --save_path ./custom_fine_tuned_model \
    --results_path ./custom_results \
    --seed 123
```

### Programmatic Usage

```python
from extension import SafetyNeuronAnalyzer, FineTuner, recalculate_safety_scores

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("llama2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("llama2-7b-chat-hf")

# Step 1: Identify safety-critical neurons
analyzer = SafetyNeuronAnalyzer(model, tokenizer)
safety_masks = analyzer.identify_safety_critical_neurons(
    prune_method="wandg",
    prune_data="align",
    sparsity_ratio=0.1
)

# Step 2: Freeze neurons and fine-tune
model = analyzer.freeze_safety_critical_neurons(model)
fine_tuner = FineTuner(model, tokenizer)
fine_tuned_model = fine_tuner.fine_tune_model(
    training_data="alpaca_cleaned_no_safety",
    num_epochs=3
)

# Step 3: Recalculate scores
new_safety_masks = recalculate_safety_scores(
    fine_tuned_model, tokenizer, analyzer
)
```

## Command Line Arguments

### Model Arguments
- `--model`: Model name to analyze (default: `llama2-7b-chat-hf`)
- `--cache_dir`: Cache directory for models (default: `llm_weights`)

### Safety Analysis Arguments
- `--prune_method`: Method for identifying safety-critical neurons (`wandg` or `wanda`, default: `wandg`)
- `--prune_data`: Dataset for safety analysis (`align` or `align_short`, default: `align`)
- `--sparsity_ratio`: Fraction of neurons to identify as safety-critical (default: `0.1`)
- `--nsamples`: Number of samples for safety analysis (default: `128`)

### Fine-tuning Arguments
- `--training_data`: Dataset for fine-tuning (default: `alpaca_cleaned_no_safety`)
- `--num_epochs`: Number of training epochs (default: `3`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--batch_size`: Batch size (default: `4`)
- `--max_length`: Maximum sequence length (default: `512`)

### Output Arguments
- `--save_path`: Path to save fine-tuned model (default: `./fine_tuned_model`)
- `--results_path`: Path to save analysis results (default: `./results`)

### Other Arguments
- `--seed`: Random seed (default: `42`)
- `--device`: Device to use (default: `cuda:0`)

## Output Files

The extension generates several output files:

### Model Files
- `{save_path}/`: Directory containing the fine-tuned model and tokenizer
- `pytorch_model.bin`: Fine-tuned model weights
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer files

### Analysis Results
- `{results_path}/original_safety_masks.pt`: PyTorch tensor containing original safety-critical neuron masks
- `{results_path}/new_safety_masks.pt`: PyTorch tensor containing new safety-critical neuron masks after fine-tuning
- `{results_path}/analysis_summary.json`: JSON file with quantitative analysis summary

### Analysis Summary Format
```json
{
  "original_safety_critical_count": 1234567,
  "new_safety_critical_count": 987654,
  "total_neurons": 12345678,
  "prune_method": "wandg",
  "prune_data": "align",
  "sparsity_ratio": 0.1,
  "training_data": "alpaca_cleaned_no_safety",
  "num_epochs": 3,
  "learning_rate": 2e-5
}
```

## Research Applications

### 1. Safety Alignment Analysis
- Study how safety-critical neurons change during fine-tuning
- Analyze the stability of safety mechanisms under different training conditions
- Investigate the relationship between utility and safety neurons

### 2. Robustness Studies
- Test model robustness when safety-critical neurons are frozen
- Compare fine-tuning behavior with and without neuron freezing
- Analyze the trade-offs between safety preservation and model adaptation

### 3. Interpretability Research
- Understand which neurons are responsible for safety behaviors
- Study the evolution of safety-critical patterns during training
- Investigate the relationship between neuron importance and model behavior

## Technical Considerations

### Memory Requirements
- The extension requires sufficient GPU memory to load the model and perform fine-tuning
- Consider using gradient checkpointing for larger models
- The freezing mechanism adds minimal memory overhead

### Computational Complexity
- Safety neuron identification scales with model size and dataset size
- Fine-tuning complexity depends on the number of epochs and dataset size
- Score recalculation requires another pass through the safety dataset

### Reproducibility
- All random operations use the specified seed for reproducibility
- Model loading and saving preserve exact weights
- Analysis results are deterministic given the same inputs

## Limitations and Future Work

### Current Limitations
1. **Binary Freezing**: Currently implements binary freezing (freeze/not freeze) rather than partial freezing
2. **Static Analysis**: Safety-critical neurons are identified once and not updated during fine-tuning
3. **Limited Datasets**: Currently supports a limited set of safety and training datasets

### Potential Extensions
1. **Dynamic Freezing**: Implement adaptive freezing that updates during training
2. **Partial Freezing**: Allow partial updates to safety-critical neurons
3. **Multi-objective Training**: Incorporate safety loss terms during fine-tuning
4. **Layer-wise Analysis**: Provide more granular analysis of different model layers

## Dependencies

The extension requires the following packages:
- `torch`: PyTorch framework
- `transformers`: HuggingFace Transformers library
- `datasets`: HuggingFace Datasets library
- `numpy`: Numerical computing
- `json`: JSON handling
- `argparse`: Command-line argument parsing

All dependencies are included in the existing `requirements.txt` file.

## Citation

If you use this extension in your research, please cite the original paper:

```bibtex
@article{wei2024assessing,
  title={Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications},
  author={Wei, Boyi and Huang, Kaixuan and Huang, Yangsibo and Xie, Tinghao and Qi, Xiangyu and Xia, Mengzhou and Mittal, Prateek and Wang, Mengdi and Henderson, Peter},
  journal={arXiv preprint arXiv:2402.05162},
  year={2024}
}
```

