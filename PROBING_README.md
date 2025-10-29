# Probing Method for Identifying Safety-Critical Attention Heads

This document explains how to use the probing method to identify safety-critical attention heads in alignment-trained language models.

## Overview

The probing method identifies which attention heads are most important for distinguishing between harmful and harmless instructions. Following the approach described in Appendix B of the paper, we:

1. Feed the model with both harmful and harmless instructions
2. Collect activation outputs from each attention head
3. Train a linear classifier for each head to distinguish harmful vs harmless activations
4. Rank heads by classification accuracy on a validation set

## Method Details

### Data
- **Harmful instructions**: 420 samples from AdvBenchattr (`data/advbench.txt`)
- **Harmless instructions**: 420 randomly sampled instructions from the utility dataset (`data/alpaca_cleaned_no_safety_train.csv`)

### Train/Validation Split
- Data is split with a 5:2 ratio (train:validation)
- Default: 600 training samples, 240 validation samples

### Classifier
- Linear classifier (Logistic Regression) trained for each attention head
- Uses activation outputs as features to predict harmful (1) vs harmless (0)

### Activation Collection
- Activations are extracted from the input to the output projection (`o_proj`) of each attention layer
- This captures the concatenated head outputs before they are combined
- Each head's activation is mean-pooled over the sequence length for a fixed-size representation

## Usage

### Test Mode (Quick Verification)

First, verify the implementation works with a small sample:

```bash
python probing.py \
    --model llama2-7b-chat-hf \
    --model_path models/llama-2-7b-chat-hf/ \
    --test_mode
```

This will run with only 10 harmful + 10 harmless instructions (20 total samples) to quickly verify everything works.

### Basic Usage

Run the probing script with default parameters (420 harmful + 420 harmless):

```bash
python probing.py \
    --model llama2-7b-chat-hf \
    --model_path models/llama-2-7b-chat-hf/ \
    --harmful_data data/advbench.txt \
    --harmless_data data/alpaca_cleaned_no_safety_train.csv \
    --output_file data/probing_result_7b.json
```

### For Llama-2-13B

```bash
python probing.py \
    --model llama2-13b-chat-hf \
    --model_path models/llama-2-13b-chat-hf/ \
    --output_file data/probing_result_13b.json
```

### Advanced Options

```bash
python probing.py \
    --model llama2-7b-chat-hf \
    --model_path models/llama-2-7b-chat-hf/ \
    --harmful_data data/advbench.txt \
    --harmless_data data/alpaca_cleaned_no_safety_train.csv \
    --n_samples 420 \
    --batch_size 8 \
    --max_length 512 \
    --seed 42 \
    --train_split 0.714286 \
    --output_file data/probing_result_7b.json
```

### Parameters

- `--model`: Model name (default: `llama2-7b-chat-hf`)
- `--model_path`: Path to model weights
- `--harmful_data`: Path to harmful instructions file (default: `data/advbench.txt`)
- `--harmless_data`: Path to harmless instructions CSV (default: `data/alpaca_cleaned_no_safety_train.csv`)
- `--n_samples`: Number of samples per class (default: 420)
- `--batch_size`: Batch size for inference (default: 8)
- `--max_length`: Maximum sequence length (default: 512)
- `--seed`: Random seed for reproducibility (default: 42)
- `--train_split`: Ratio of data for training (default: 5/7 ≈ 0.714286)
- `--output_file`: Output JSON file for results (default: `data/probing_result_7b.json`)

## Output

The script produces a JSON file with the following format:

```json
{
  "layer-0-head-0": 0.9583,
  "layer-0-head-1": 0.9750,
  ...
  "layer-31-head-31": 0.9916
}
```

Each key is an attention head identifier (`layer-{layer_idx}-head-{head_idx}`), and the value is the validation accuracy (0.0 to 1.0) of the linear classifier for that head.

### Interpretation

- **Higher accuracy** indicates the head is more important for distinguishing harmful vs harmless instructions
- Heads with accuracy close to 1.0 are strongly safety-critical
- Heads with accuracy close to 0.5 are not useful for distinguishing the two classes

## Integration with Pruning

The generated probing results can be used with the attention head pruning method:

```bash
python main.py \
    --model llama2-7b-chat-hf \
    --prune_method attention_head \
    --sparsity_ratio 0.1 \
    --eval_attack
```

This will:
1. Load the probing results from `data/probing_result_7b.json`
2. Identify the top-k most safety-critical heads based on accuracy
3. Prune (zero out) those heads
4. Evaluate the attack success rate

## Example Output

During execution, you'll see:

```
Loading model from models/llama-2-7b-chat-hf/...
Model has 32 layers and 32 attention heads per layer
Hidden dimension: 4096, Head dimension: 128

Loading 420 harmful instructions...
Loaded 420 harmful instructions

Loading 420 harmless instructions...
Loaded 420 harmless instructions

Train set: 600 samples
Validation set: 240 samples

Collecting training activations...
100%|████████████████| 75/75 [02:30<00:00,  2.01s/it]

Collecting validation activations...
100%|████████████████| 30/30 [01:00<00:00,  2.00s/it]

Training linear classifiers for 1024 attention heads...
100%|████████████████| 1024/1024 [05:20<00:00,  3.20it/s]

================================================================================
Top 10 safety-critical attention heads:
================================================================================
 1. layer-15-head-23        : 1.0000
 2. layer-8-head-12         : 1.0000
 3. layer-31-head-0         : 1.0000
 4. layer-7-head-0          : 1.0000
 5. layer-22-head-2         : 1.0000
 6. layer-16-head-0         : 1.0000
 7. layer-1-head-0          : 1.0000
 8. layer-9-head-0          : 1.0000
 9. layer-31-head-4         : 1.0000
10. layer-5-head-0          : 1.0000

================================================================================
Summary Statistics:
================================================================================
Mean accuracy: 0.9654
Std accuracy:  0.0312
Min accuracy:  0.7792
Max accuracy:  1.0000
Median accuracy: 0.9750
```

## Requirements

- PyTorch
- Transformers
- scikit-learn
- pandas
- numpy
- tqdm

Install dependencies:
```bash
pip install torch transformers scikit-learn pandas numpy tqdm
```

## Notes

- The script requires significant GPU memory (recommended: 24GB+ for 7B model, 40GB+ for 13B model)
- Processing time depends on batch size and hardware (typically 10-20 minutes for 840 samples)
- Results are deterministic when using the same seed
- The script automatically handles padding and truncation for variable-length inputs

## Comparison with Existing Results

You can compare your generated results with the pre-computed results in:
- `data/probing_result_7b.json` (for Llama-2-7B-chat)
- `data/probing_result_13b.json` (for Llama-2-13B-chat)

To verify your implementation:
```python
import json

# Load your results
with open('data/my_probing_result.json', 'r') as f:
    my_results = json.load(f)

# Load reference results
with open('data/probing_result_7b.json', 'r') as f:
    ref_results = json.load(f)

# Compare (should be similar, minor variations due to random sampling)
import numpy as np
my_accuracies = list(my_results.values())
ref_accuracies = list(ref_results.values())

print(f"My mean accuracy: {np.mean(my_accuracies):.4f}")
print(f"Reference mean accuracy: {np.mean(ref_accuracies):.4f}")
```

## Citation

If you use this probing method, please cite the original paper:

```bibtex
@article{zhu2024assessing,
  title={Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications},
  author={Zhu, Boyi and others},
  journal={arXiv preprint arXiv:2402.05162},
  year={2024}
}
```

And the probing methodology papers:

```bibtex
@inproceedings{hewitt2019designing,
  title={Designing and Interpreting Probes with Control Tasks},
  author={Hewitt, John and Liang, Percy},
  booktitle={EMNLP},
  year={2019}
}
```
