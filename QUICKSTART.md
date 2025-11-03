# Quick Start Guide 

## Setup Check

Before running, verify your environment:

```bash
cd /workspace/CS2881-alignment-attribution-code
bash scripts/verify_setup.sh
```

This checks:
- Python packages (torch, transformers, peft, datasets, scipy, matplotlib)
- Disk space (/dev/shm needs ~15GB, /workspace needs ~10GB)
- GPU availability
- Model and data files
- Script permissions

## Quick Run (Full Experiment)

### Neuron Overlap Experiment 

```bash
cd /workspace/CS2881-alignment-attribution-code
bash scripts/run_full_experiment_1.sh
```

**Time**: 3-4 hours on A100
**Output**: Results in `outputs/` directory

### Neuron Drift Experiment

```bash
cd /workspace/CS2881-alignment-attribution-code
bash scripts/run_full_experiment_2.sh
```

**Time**: 6-12 hours on H100
**Output**: Results in `results/` directory

## Step-by-Step Run

### Phase 1a: Compute SNIP Scores (~0.5-1 hour)

```bash
bash scripts/phase1_compute_snip_scores.sh
```

Computes neuron importance scores on safety and utility datasets.

### Phase 1b: Identify Neuron Groups (~10-20 minutes)

```bash
bash scripts/phase1_identify_neurons.sh
```

Extracts 4 neuron groups: safety (SNIP top), safety (set diff), utility, random.

### Phase 2: Fine-Tune with Tracking (~1.5 hours)

```bash
bash scripts/phase2_finetune.sh
```

Fine-tunes with LoRA while freezing safety-critical neurons (exp 1) or tracking neuron weight drift (exp 2).

### Experiment 1: 

#### Phase 3: Identify new neuron groups (10-20 minutes)
```bash
bash scripts/phase3_get_set_diff_for_finetuned_model.sh 
```

Get top safety, top utility and safety-critical neuron groups from finetuned model. 

#### Phase 4: Evaluate results (10-20 minutes) 

```bash 
python eval_results.py 
```

### Experiment 2: 

#### Phase 3: Analyze Results (~5-10 minutes)

```bash
python analyze_drift.py
```

Generates plots, statistics, and summary report.

## Quick Test (Fast Mode)

For testing the pipeline without waiting hours:

Edit `scripts/phase2_finetune.sh`:
```bash
MAX_STEPS=100  # Instead of -1 (full training)
```

Then run:
```bash
bash scripts/phase1_compute_snip_scores.sh  # Still takes 2-4 hours
bash scripts/phase1_identify_neurons.sh      # Fast
bash scripts/phase2_finetune.sh              # Now only ~10-15 minutes
python analyze_drift.py                      # Fast
```

## View Results

After completion, check:

```bash
# Summary report
cat results/drift_analysis_report.txt

# Plots
ls results/figures/

# Statistical details
cat results/statistical_results.json
```

## Key Files

- **Input**: Pre-trained LLaMA-2-7B-Chat model
- **Intermediate**:
  - SNIP scores: `/dev/shm/snip_scores/` (can delete after Phase 1b)
  - Neuron groups: `neuron_groups/*.json` (keep)
  - Drift logs: `/dev/shm/drift_logs/` (can delete after Phase 3)
- **Output**:
  - Fine-tuned model: `finetuned_models/`
  - Analysis: `results/`

## Cleanup

Free up space after experiment:

```bash
# Remove large temporary files (safe to delete)
rm -rf /dev/shm/snip_scores/
rm -rf /dev/shm/drift_logs/
rm -rf /dev/shm/initial_weights/

# Keep these
# - neuron_groups/ (small, needed for reproduction)
# - finetuned_models/ (if you want to run more experiments)
# - results/ (your findings!)
```

## Troubleshooting

### Out of GPU Memory
```bash
# Edit scripts/phase2_finetune.sh
BATCH_SIZE=2      # Reduce from 4
GRAD_ACCUM=8      # Increase from 4
```

### Out of Disk Space
```bash
# Check space
df -h /dev/shm
df -h /workspace

# Clean up old runs
rm -rf /dev/shm/snip_scores/
```

### Model Not Found
```bash
# Update model path in scripts/phase2_finetune.sh
MODEL_PATH="/your/path/to/llama-2-7b-chat-hf/"
```

## Understanding Output

### Neuron Groups
- `neuron_groups_snip_top.json`: Top 1% SNIP on safety → Safety-critical neurons
- `neuron_groups_set_diff.json`: Safety-specific (not utility) → Disentangled safety
- `neuron_groups_utility.json`: Top 1% SNIP on utility → Utility-critical neurons
- `neuron_groups_random.json`: Random sample → Baseline

### Drift Metrics
- **Cosine Similarity**: 1.0 = no change, lower = more drift
- **L2 Distance**: 0.0 = no change, higher = more drift
- **Relative Change**: L2 distance normalized by weight magnitude

### Hypothesis Testing
- **H1 (Fragile Safety)**: Safety neurons drift MORE than random
  - → Lower cosine similarity, higher L2 distance
- **H2 (Pathway Creation)**: Safety neurons drift SIMILAR to random
  - → Suggests new harmful circuits, not neuron modification
  - → Aligns with paper's finding that freezing doesn't help

## Advanced Usage

### Different Set Difference Parameters

Try different p, q values (paper used 0.1 to 0.9):

```bash
# Edit scripts/phase1_identify_neurons.sh
SET_DIFF_P=0.3  # Top 30% utility
SET_DIFF_Q=0.5  # Top 50% safety

# Re-run Phase 1b and Phase 2
bash scripts/phase1_identify_neurons.sh
bash scripts/phase2_finetune.sh
```

### More Frequent Drift Tracking

```bash
# Edit scripts/phase2_finetune.sh
DRIFT_LOG_INTERVAL=50  # Every 50 steps instead of 100

# More data points, but larger logs
```

### Stronger LoRA Adaptation

```bash
# Edit scripts/phase2_finetune.sh
LORA_R=16         # Increase rank
LORA_ALPHA=32     # Increase alpha

# More parameters, stronger fine-tuning effect
```

## Citation

If you use this code, cite the original paper:

Wei et al. (2024). "Assessing the Brittleness of Safety Alignment via Pruning
and Low-Rank Modifications." ICML 2024.

## Documentation

- Original paper README: [README.md](README.md)
