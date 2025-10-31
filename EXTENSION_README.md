# Safety-Critical Neuron Drift Extension

## Overview

This extension investigates **why safety alignment is brittle** by measuring how safety-critical neurons change during LoRA fine-tuning.

**Research Question**: Do safety-critical neurons drift more than other neurons during fine-tuning?

## Quick Start

```bash
# 1. Verify setup
bash scripts/verify_setup.sh

# 2. Run full experiment (6-12 hours)
bash scripts/run_full_experiment.sh

# 3. View results
cat results/drift_analysis_report.txt
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## What This Does

1. **Identifies neuron groups** using SNIP scores:
   - Safety-critical neurons (SNIP top-1% on safety data)
   - Safety-critical neurons (Set difference method)
   - Utility-critical neurons (SNIP top-1% on utility data)
   - Random neurons (baseline)

2. **Fine-tunes with LoRA** while tracking drift:
   - Applies LoRA to LLaMA-2-7B-Chat
   - Fine-tunes on Alpaca dataset
   - Measures neuron weight changes every 100 steps

3. **Analyzes drift patterns**:
   - Computes cosine similarity, L2 distance, relative change
   - Compares safety-critical vs random neurons
   - Tests hypotheses about safety brittleness

## Directory Structure

```
scripts/                    # Bash scripts for each phase
lib/drift_utils.py          # Drift computation utilities
identify_neuron_groups.py   # Neuron group extraction
finetune_with_tracking.py   # LoRA fine-tuning with tracking
analyze_drift.py            # Analysis and visualization

neuron_groups/              # Output: Neuron IDs by group
finetuned_models/           # Output: Fine-tuned model
results/                    # Output: Analysis results
```

## Files Created

### Scripts (5 files, all executable)
- `scripts/phase1_compute_snip_scores.sh` - Compute SNIP scores
- `scripts/phase1_identify_neurons.sh` - Extract neuron groups
- `scripts/phase2_finetune.sh` - Fine-tune with tracking
- `scripts/run_full_experiment.sh` - Master script
- `scripts/verify_setup.sh` - Setup verification

### Python Modules (4 files)
- `lib/drift_utils.py` - Drift computation (9.2K)
- `identify_neuron_groups.py` - Neuron extraction (10K)
- `finetune_with_tracking.py` - Fine-tuning (11K)
- `analyze_drift.py` - Analysis (15K)

### Documentation (3 files)
- `QUICKSTART.md` - Quick start guide
- `docs/EXTENSION_IMPLEMENTATION.md` - Full implementation guide
- `IMPLEMENTATION_SUMMARY.md` - What was created

## Storage Management

**Large temporary files** → `/dev/shm` (117 GB):
- SNIP scores (~5-10 GB)
- Drift logs (~2-5 GB)
- Initial weights (~1-2 GB)

**Final outputs** → `/workspace` (50 GB):
- Neuron groups (small JSON files)
- Fine-tuned model (~7 GB)
- Analysis results (small)

**Only neuron IDs are saved, not SNIP scores** (per requirement).

## Usage

### Full Experiment
```bash
bash scripts/run_full_experiment.sh
```

### Individual Phases
```bash
# Phase 1a: Compute SNIP scores (~2-4 hours)
bash scripts/phase1_compute_snip_scores.sh

# Phase 1b: Identify neuron groups (~5-10 minutes)
bash scripts/phase1_identify_neurons.sh

# Phase 2: Fine-tune with tracking (~4-8 hours)
bash scripts/phase2_finetune.sh

# Phase 3: Analyze results (~5-10 minutes)
python analyze_drift.py
```

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

## Cleanup

After experiment, free up space:
```bash
# Remove large temporary files (safe)
rm -rf /dev/shm/snip_scores/
rm -rf /dev/shm/drift_logs/
rm -rf /dev/shm/initial_weights/

# Keep: neuron_groups/, finetuned_models/, results/
```

## Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Full Guide**: [docs/EXTENSION_IMPLEMENTATION.md](docs/EXTENSION_IMPLEMENTATION.md)
- **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Original Paper**: [README.md](README.md)

## Implementation Notes

### Code Reuse
- SNIP computation: Uses existing `main.py --dump_wanda_score`
- Data loading: Uses existing `lib/data.py`
- Model loading: Uses same patterns as original code

### New Components
- All new code is in separate files
- No modifications to existing codebase
- Clean, modular design (~350 lines for fine-tuning)

### Design Choices
- **LoRA**: Efficient, still degrades safety (per paper)
- **Per-neuron tracking**: Granular analysis
- **Storage optimization**: Large files in /dev/shm
- **Simple implementation**: Standard PEFT library

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

## Status

✅ **Implementation complete and ready to run**

All code has been created but **not executed**. You can now:
1. Verify setup: `bash scripts/verify_setup.sh`
2. Run experiment: `bash scripts/run_full_experiment.sh`
3. Analyze results: Review `results/` directory

No files have been modified in the original codebase.
