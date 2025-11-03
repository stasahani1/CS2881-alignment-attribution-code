# Assessing the Brittleness of Safety Alignment — Safety-Critical Neuron Overlap and Neuron Drift Extension

## Overview

This extension investigates **why safety alignment is brittle** by measuring how safety-critical neurons change during LoRA fine-tuning. The results of the investigation is summarised in this [report](CS2881_Mini_Project.pdf). 

**Research Questions**: 
- Does model create alternative pathway during finetuning with frozen safety-critical neurons? 
- Do safety-critical neurons drift more than other neurons during fine-tuning?

## Quick Start 
See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

### Neuron Overlap Analysis 

```bash
# 1. Verify setup
bash scripts/verify_setup.sh

# 2. Run full experiment (3-4 hours)
bash scripts/run_full_experiment_1.sh
``` 

### Neuron Drift Analysis 

```bash
# 1. Verify setup
bash scripts/verify_setup.sh

# 2. Run full experiment (6-12 hours)
bash scripts/run_full_experiment_2.sh

# 3. View results
cat results/drift_analysis_report.txt
```

## What This Does

1. **Identifies neuron groups** using SNIP scores:
   - Top safety neurons (SNIP top-1% on safety data)
   - Safety-critical neurons (Set difference method)
   - Top utility neurons (SNIP top-1% on utility data)
   - Random neurons (baseline)

2. **Fine-tunes with LoRA** while tracking drift:
   - Applies LoRA to LLaMA-2-7B-Chat
   - Fine-tunes on Alpaca dataset (Optional: freeze safety-critical neurons)
   - Measures neuron weight changes every 100 steps

3. **(Experiment 1) Identify new neuron groups**: 
   - Identify new top safety, top utility, and safety-critical neurons after finetuning 
   - Evaluate neuron overlap (see [eval_results.py](eval_results.py))

4. **(Experiment 2) Analyzes drift patterns**:
   - Computes cosine similarity, L2 distance, relative change
   - Compares safety-critical vs random neurons
   - Tests hypotheses about safety brittleness

### Quick Test
For testing without waiting hours, edit `scripts/phase2_finetune.sh`:
```bash
MAX_STEPS=100  # Run only 100 steps instead of full training
```

## Results

After completion, you'll get:

1. **Text Report**: `results/drift_analysis_report.txt`
   - Final drift metrics by group
   - Statistical comparisons (Cohen's d)
   - Hypothesis evaluation

2. **Plots**: `results/figures/`
   - Time series: drift over training
   - Distributions: final drift by group

3. **Statistics**: `results/statistical_results.json`
   - Detailed statistical tests

## Hypotheses

### Experiment 1: 
After finetuning with frozen safety-critical neurons, the new set of safe-critical neurons for the finetuned models should be very different from the original neurons. 

### Experiment 2: 
**H1 (Fragile Safety)**: Safety neurons drift MORE than random
- → Lower cosine similarity, higher L2 distance
- → Safety mechanisms are inherently unstable

**H2 (Pathway Creation)**: Safety neurons drift SIMILAR to random
- → Cosine similarity and L2 distance similar to random
- → Fine-tuning creates new harmful circuits
- → Aligns with paper's finding that freezing doesn't help

## Configuration

All parameters are configurable via bash scripts:

**Neuron Selection** (`scripts/phase1_identify_neurons.sh`):
```bash
SNIP_TOP_K=0.01      # Top 1%
SET_DIFF_P=0.1       # Set difference p
SET_DIFF_Q=0.1       # Set difference q
```

**LoRA** (`scripts/phase2_finetune.sh`):
```bash
LORA_R=8             # Rank
LORA_ALPHA=16        # Alpha
```

**Training** (`scripts/phase2_finetune.sh`):
```bash
NUM_EPOCHS=1              # Epochs
BATCH_SIZE=4              # Batch size
LEARNING_RATE=1e-4        # Learning rate
DRIFT_LOG_INTERVAL=100    # Tracking frequency
```

## Requirements

- PyTorch
- Transformers
- PEFT (for LoRA)
- Datasets
- NumPy, SciPy, Matplotlib

Check with: `bash scripts/verify_setup.sh`

## Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Original Paper**: [ORIGINAL_README.md](ORIGINAL_README.md)

## Citation

If you use this code, cite the original paper:

```bibtex
@InProceedings{pmlr-v235-wei24f,
  title = {Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications},
  author = {Wei, Boyi and Huang, Kaixuan and Huang, Yangsibo and Xie, Tinghao and
            Qi, Xiangyu and Xia, Mengzhou and Mittal, Prateek and Wang, Mengdi and Henderson, Peter},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024},
}
```
