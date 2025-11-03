# Safety-Critical Neuron Drift Experiment - Implementation Guide

## Overview

This implementation extends the original alignment attribution codebase to investigate **why safety alignment is brittle** by measuring how safety-critical neurons change during LoRA fine-tuning.

## Research Question

**Do safety-critical neurons drift more than other neurons during fine-tuning?**

- **Hypothesis A (Fragile Safety)**: Safety neurons drift MORE → safety mechanisms are inherently unstable
- **Hypothesis B (Pathway Creation)**: Safety neurons drift LESS or SIMILAR → fine-tuning creates alternative harmful circuits

## File Structure

```
CS2881-alignment-attribution-code/
├── scripts/
│   ├── phase1_compute_snip_scores.sh    # Compute SNIP scores on safety & utility datasets
│   ├── phase1_identify_neurons.sh       # Extract neuron groups from SNIP scores
│   ├── phase2_finetune.sh              # Fine-tune with LoRA and track drift
│   └── run_full_experiment.sh          # Master script to run everything
│
├── lib/
│   └── drift_utils.py                  # NEW: Drift computation utilities
│
├── identify_neuron_groups.py           # NEW: Extract neuron IDs from SNIP scores
├── finetune_with_tracking.py          # NEW: LoRA fine-tuning with drift tracking
├── analyze_drift.py                   # NEW: Statistical analysis and visualization
│
├── neuron_groups/                     # Created by Phase 1b
│   ├── neuron_groups_snip_top.json   # Top 1% SNIP on safety data
│   ├── neuron_groups_set_diff.json   # Set difference method
│   ├── neuron_groups_utility.json    # Top 1% SNIP on utility data
│   └── neuron_groups_random.json     # Random baseline
│
├── finetuned_models/                 # Created by Phase 2
│   └── ...                           # LoRA-finetuned model
│
└── results/                          # Created by Phase 3
    ├── drift_analysis_report.txt     # Text summary of findings
    ├── statistical_results.json      # Detailed statistics
    └── figures/                      # Visualizations
        ├── cosine_similarity_mean_over_time.png
        ├── l2_distance_mean_over_time.png
        └── ...
```

## Storage Management

The implementation carefully manages storage to avoid quota issues:

- **Large temporary files** → `/dev/shm` (117 GB fast shared memory):
  - SNIP scores (~several GB)
  - Initial weight snapshots
  - Drift logs during training

- **Final outputs** → `/workspace` (50 GB):
  - Neuron group IDs (JSON, small)
  - Fine-tuned model (~7 GB)
  - Analysis results and plots

- **Key optimization**: Only neuron IDs are saved, NOT the SNIP scores themselves (per requirement #2)

## Running the Experiment

### Option 1: Run Full Experiment (Automated)

```bash
cd /workspace/CS2881-alignment-attribution-code
bash scripts/run_full_experiment.sh
```

This runs all phases sequentially. **Estimated time: 6-12 hours** on H100.

### Option 2: Run Phases Individually

#### Phase 1a: Compute SNIP Scores (~2-4 hours)

```bash
cd /workspace/CS2881-alignment-attribution-code
bash scripts/phase1_compute_snip_scores.sh
```

This computes SNIP scores for:
- Safety dataset: `align` (safety-full from paper)
- Utility dataset: `alpaca_cleaned_no_safety`

**Output**: `/dev/shm/snip_scores/align/` and `/dev/shm/snip_scores/alpaca_cleaned_no_safety/`

#### Phase 1b: Identify Neuron Groups (~5-10 minutes)

```bash
bash scripts/phase1_identify_neurons.sh
```

This extracts 4 neuron groups:
1. **Safety-critical (SNIP top-1%)**: Top 1% SNIP scores on safety data
2. **Safety-critical (Set difference)**: Top-q% safety BUT NOT in top-p% utility
3. **Utility-critical**: Top 1% SNIP scores on utility data
4. **Random baseline**: Random 1% sample

**Output**: `neuron_groups/*.json`

#### Phase 2: Fine-Tune with Drift Tracking (~4-8 hours)

```bash
bash scripts/phase2_finetune.sh
```

This:
- Loads LLaMA-2-7B-Chat model
- Applies LoRA (rank=8, alpha=16)
- Fine-tunes on Alpaca dataset
- Saves checkpoints every 500 steps

**Output**:
- Model checkpoints: `/workspace/CS2881-alignment-attribution-code/finetuned_models/`

#### Phase 2b: Analyze Drift Patterns (~5-10 minutes)

```bash
bash scripts/phase2b_compute_drift.sh
```

This:
- Loads all checkpoints
- Computes drift

**Output**:
- Drift logs: `/dev/shm/drift_logs/`



#### Phase 3: 
```
python analyze_drift.py \
    --drift_log_dir /dev/shm/drift_logs \
    --output_dir results
```

- Generates time series plots
- Computes statistical comparisons (Cohen's d effect sizes)

**Output**: `results/` directory with report and figures

## Configuration

### Neuron Group Parameters

Edit `scripts/phase1_identify_neurons.sh`:

```bash
SNIP_TOP_K=0.01      # Top 1% for SNIP method
SET_DIFF_P=0.1       # Top-p% utility (for set difference)
SET_DIFF_Q=0.1       # Top-q% safety (for set difference)
```

The paper used p and q ranging from 0.1 to 0.9. Start with 0.1, 0.1.

### LoRA Parameters

Edit `scripts/phase2_finetune.sh`:

```bash
LORA_R=8             # LoRA rank (higher = more parameters)
LORA_ALPHA=16        # LoRA scaling factor
```

Standard LoRA settings: r=8, alpha=16. Can increase for stronger adaptation.

### Training Parameters

Edit `scripts/phase2_finetune.sh`:

```bash
NUM_EPOCHS=1              # Number of epochs
BATCH_SIZE=4              # Batch size per GPU
GRAD_ACCUM=4              # Gradient accumulation steps
LEARNING_RATE=1e-4        # Learning rate
MAX_STEPS=-1              # Max steps (-1 = full training)
DRIFT_LOG_INTERVAL=100    # Steps between drift measurements
```

For quick testing, set `MAX_STEPS=500` to run only 500 steps.

## Understanding the Output

### Neuron Group Files

Each JSON file maps layer names to neuron coordinates:

```json
{
  "layer_0_self_attn.q_proj": [[245, 1023], [891, 2048], ...],
  "layer_1_mlp.down_proj": [[12, 456], ...]
}
```

Format: `[row, col]` where row=output neuron index, col=input dimension index.

### Drift Logs

Each checkpoint creates a JSON file:

```json
{
  "step": 100,
  "metrics": {
    "cosine_similarity": {
      "safety_snip_top": {"mean": 0.9987, "std": 0.0012, ...},
      "safety_set_diff": {"mean": 0.9985, "std": 0.0015, ...},
      "utility": {"mean": 0.9990, "std": 0.0010, ...},
      "random": {"mean": 0.9989, "std": 0.0011, ...}
    },
    "l2_distance": {...},
    "relative_change": {...}
  }
}
```

### Drift Metrics

1. **Cosine Similarity**: Measures direction change
   - 1.0 = no change in direction
   - 0.0 = orthogonal
   - -1.0 = opposite direction
   - **Lower values = more drift**

2. **L2 Distance**: Measures magnitude of change
   - 0.0 = no change
   - **Higher values = more drift**

3. **Relative Change**: L2 distance normalized by initial norm
   - Accounts for different weight scales across neurons
   - **Higher values = more drift**

### Analysis Report

The report (`results/drift_analysis_report.txt`) includes:

1. **Final values by group**: Mean, std, median, range for each metric
2. **Statistical comparisons**: Cohen's d effect sizes
3. **Hypothesis evaluation**: Which hypothesis is supported by data

**Interpreting Cohen's d**:
- |d| < 0.2: Negligible effect
- |d| < 0.5: Small effect
- |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect

### Visualizations

Generated plots in `results/figures/`:

1. **Time series plots**: Show how metrics change over training
   - `*_mean_over_time.png`: Mean drift by group
   - `*_std_over_time.png`: Variability over time

2. **Distribution plots**: Show final metric distributions
   - `*_final_distribution.png`: Box plots comparing groups

## Expected Results

Based on the paper's findings, you might observe:

### If Hypothesis A is True (Fragile Safety):
- Safety neurons have **lower cosine similarity** than random
- Safety neurons have **higher L2 distance** than random
- Large Cohen's d (> 0.5) between safety and random groups
- **Interpretation**: Safety alignment is in fragile neurons that change easily

### If Hypothesis B is True (Pathway Creation):
- Safety neurons have **similar or higher cosine similarity** than random
- Safety neurons have **similar or lower L2 distance** than random
- Small Cohen's d (< 0.2) between safety and random groups
- **Interpretation**: Safety degradation comes from new harmful circuits, not safety neuron modification
- **This would align with the paper's finding** that freezing <50% safety neurons doesn't help

## Cleanup

After analysis, you can free up space by removing temporary files:

```bash
# Remove SNIP scores (largest files)
rm -rf /dev/shm/snip_scores/

# Remove drift logs
rm -rf /dev/shm/drift_logs/

# Remove initial weights
rm -rf /dev/shm/initial_weights/

# Keep these for future reference:
# - neuron_groups/*.json (small)
# - finetuned_models/ (needed if you want to run more evals)
# - results/ (your findings!)
```

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size in `scripts/phase2_finetune.sh`:
```bash
BATCH_SIZE=2          # Instead of 4
GRAD_ACCUM=8          # Instead of 4 (keeps effective batch size constant)
```

### Out of Disk Space

The implementation already uses `/dev/shm` for large files. If still running out:

1. Check available space:
   ```bash
   df -h /dev/shm
   df -h /workspace
   ```

2. Clean up old files:
   ```bash
   rm -rf /dev/shm/snip_scores/
   ```

3. Reduce drift logging frequency:
   ```bash
   DRIFT_LOG_INTERVAL=500  # Instead of 100
   ```

### Model Path Not Found

Update model path in `scripts/phase2_finetune.sh`:
```bash
MODEL_PATH="/your/path/to/llama-2-7b-chat-hf/"
```

### SNIP Score Computation Fails

Make sure datasets are available in `data/`:
- `SFT_aligned_llama2-7b-chat-hf_train.csv`
- Alpaca dataset (loaded via HuggingFace datasets library)

## Extending the Experiment

### Test Different Set Difference Parameters

Modify `scripts/phase1_identify_neurons.sh`:
```bash
SET_DIFF_P=0.3  # Try different values
SET_DIFF_Q=0.5
```

Run Phase 1b again, then Phase 2 with new neuron groups.

### Track More Checkpoints

Reduce `DRIFT_LOG_INTERVAL` to track drift more frequently:
```bash
DRIFT_LOG_INTERVAL=50  # Every 50 steps instead of 100
```

### Use Different Fine-Tuning Dataset

Modify `finetune_with_tracking.py` to load a different dataset in the `load_alpaca_dataset()` function.

## Implementation Notes

### Reused Components

The implementation maximally reuses existing code:
- `main.py --dump_wanda_score`: SNIP score computation
- `lib/data.py`: Dataset loaders
- `lib/model_wrapper.py`: SNIP implementation

### New Components

All new code is cleanly separated:
- `lib/drift_utils.py`: Drift computation (no dependencies on existing code)
- `identify_neuron_groups.py`: Standalone neuron extraction
- `finetune_with_tracking.py`: Standalone fine-tuning script
- `analyze_drift.py`: Standalone analysis script

### Design Decisions

1. **LoRA instead of full fine-tuning**: More efficient, still degrades safety (per paper)
2. **Track individual neurons, not full layers**: More granular analysis
3. **Save only neuron IDs**: Per requirement #2, minimize storage
4. **Use /dev/shm for large files**: Fast access, avoid /workspace quota
5. **Statistical tests use summary statistics**: We aggregate per-neuron metrics by group

## Citation

If you use this extension in your research, please cite the original paper:

```bibtex
@InProceedings{pmlr-v235-wei24f,
  title = {Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications},
  author = {Wei, Boyi and Huang, Kaixuan and Huang, Yangsibo and Xie, Tinghao and
            Qi, Xiangyu and Xia, Mengzhou and Mittal, Prateek and Wang, Mengdi and Henderson, Peter},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024},
}
```
