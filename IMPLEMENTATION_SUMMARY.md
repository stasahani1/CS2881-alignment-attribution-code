# Implementation Summary - Safety-Critical Neuron Drift Extension

## What Was Created

This implementation adds a complete experimental pipeline to investigate neuron drift during fine-tuning.

### New Files Created

#### Scripts (5 files)
```
scripts/
├── phase1_compute_snip_scores.sh    # Compute SNIP scores (Phase 1a)
├── phase1_identify_neurons.sh       # Extract neuron groups (Phase 1b)
├── phase2_finetune.sh              # Fine-tune with drift tracking (Phase 2)
├── run_full_experiment.sh          # Master script (all phases)
└── verify_setup.sh                 # Setup verification
```

All scripts are executable (`chmod +x`).

#### Python Modules (4 files)
```
lib/drift_utils.py                  # Drift computation utilities (~200 lines)
identify_neuron_groups.py           # Neuron group extraction (~350 lines)
finetune_with_tracking.py          # LoRA fine-tuning with tracking (~350 lines)
analyze_drift.py                   # Analysis and visualization (~450 lines)
```

#### Documentation (3 files)
```
docs/EXTENSION_IMPLEMENTATION.md    # Full implementation guide
QUICKSTART.md                       # Quick start guide
IMPLEMENTATION_SUMMARY.md          # This file
```

### Key Features

#### ✅ Reuses Existing Code
- SNIP score computation: Uses `main.py --dump_wanda_score`
- Dataset loading: Uses `lib/data.py` loaders
- Model loading: Uses same `get_llm()` pattern

#### ✅ Clean Separation
- All new code is in separate files
- No modifications to existing codebase
- Can be removed without breaking original functionality

#### ✅ Storage Optimization
- Large temporary files → `/dev/shm` (117 GB fast memory)
  - SNIP scores (~5-10 GB)
  - Drift logs (~2-5 GB)
  - Initial weights (~1-2 GB)
- Final outputs → `/workspace` (50 GB)
  - Neuron groups (small JSON files)
  - Fine-tuned model (~7 GB)
  - Analysis results (small)
- **Only neuron IDs saved, not scores** (per requirement #2)

#### ✅ Simple and Clean
- Fine-tuning code is straightforward (~350 lines total)
- Uses standard PEFT library for LoRA
- Minimal dependencies (all in requirements.txt)

### Implementation Approach

#### Phase 1: Neuron Identification
1. **Compute SNIP scores** (existing code):
   ```bash
   python main.py --dump_wanda_score --prune_method wandg --prune_data align
   ```
2. **Extract neuron groups** (new code):
   - SNIP top-k method
   - Set difference method (top-q% safety NOT in top-p% utility)
   - Random baseline

#### Phase 2: Fine-Tuning with Tracking
1. **Load model and neuron groups**
2. **Save initial weights** (before LoRA)
3. **Apply LoRA** (rank=8, alpha=16)
4. **Fine-tune on Alpaca**
5. **Track drift every N steps**:
   - Extract current weights for tracked neurons
   - Compute cosine similarity vs initial
   - Compute L2 distance vs initial
   - Compute relative change
   - Aggregate by neuron group
   - Save to log file

#### Phase 3: Analysis
1. **Load all drift logs**
2. **Generate time series plots**:
   - Cosine similarity over time
   - L2 distance over time
   - Relative change over time
3. **Statistical tests**:
   - Cohen's d effect sizes
   - Compare safety vs random groups
4. **Hypothesis evaluation**:
   - H1: Safety neurons drift MORE (fragile)
   - H2: Safety neurons drift SIMILAR (pathway creation)

### Neuron Groups Compared

1. **Safety-critical (SNIP top)**: Top 1% SNIP scores on safety dataset
2. **Safety-critical (Set difference)**: Top-q% safety NOT in top-p% utility
3. **Utility-critical**: Top 1% SNIP scores on utility dataset
4. **Random**: Random 1% sample (baseline)

### Drift Metrics

1. **Cosine Similarity**: Direction change (1.0 = no change, lower = more drift)
2. **L2 Distance**: Magnitude of change (0.0 = no change, higher = more drift)
3. **Relative Change**: Normalized by initial magnitude

### Storage Layout

```
/dev/shm/                              # Fast temporary storage (117 GB)
├── snip_scores/
│   ├── align/
│   │   └── wanda_score/
│   │       └── W_metric_*.pkl        # ~5 GB total
│   └── alpaca_cleaned_no_safety/
│       └── wanda_score/
│           └── W_metric_*.pkl        # ~5 GB total
├── drift_logs/
│   └── drift_step_*.json             # ~2-5 GB total
└── initial_weights/
    └── initial_weights.pt            # ~1-2 GB

/workspace/CS2881-alignment-attribution-code/
├── neuron_groups/
│   ├── neuron_groups_snip_top.json   # Small (<1 MB)
│   ├── neuron_groups_set_diff.json   # Small
│   ├── neuron_groups_utility.json    # Small
│   └── neuron_groups_random.json     # Small
├── finetuned_models/
│   └── ...                           # ~7 GB
└── results/
    ├── drift_analysis_report.txt     # Small
    ├── statistical_results.json      # Small
    └── figures/
        └── *.png                     # Small (~10 MB total)
```

### Running the Experiment

#### Full Pipeline (One Command)
```bash
bash scripts/run_full_experiment.sh
```

#### Individual Phases
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

#### Quick Test
```bash
# Edit scripts/phase2_finetune.sh:
MAX_STEPS=100  # Instead of -1

# Then run Phase 2 (~10-15 minutes instead of hours)
bash scripts/phase2_finetune.sh
```

### Expected Outcomes

Based on the paper's findings about safety brittleness:

#### If H1 (Fragile Safety) is true:
- Safety neurons have **lower cosine similarity** than random
- Safety neurons have **higher L2 distance** than random
- **Interpretation**: Safety is in fragile neurons that change easily

#### If H2 (Pathway Creation) is true:
- Safety neurons have **similar cosine similarity** to random
- Safety neurons have **similar L2 distance** to random
- **Interpretation**: Safety degradation comes from new harmful circuits
- **Aligns with paper**: Freezing <50% safety neurons doesn't prevent attacks

### Configuration Options

All configurable via bash scripts (no code editing needed):

#### Neuron Selection (`scripts/phase1_identify_neurons.sh`)
```bash
SNIP_TOP_K=0.01      # Top 1% for SNIP
SET_DIFF_P=0.1       # Top 10% utility
SET_DIFF_Q=0.1       # Top 10% safety
```

#### LoRA Parameters (`scripts/phase2_finetune.sh`)
```bash
LORA_R=8             # Rank
LORA_ALPHA=16        # Alpha
```

#### Training (`scripts/phase2_finetune.sh`)
```bash
NUM_EPOCHS=1              # Epochs
BATCH_SIZE=4              # Batch size
LEARNING_RATE=1e-4        # Learning rate
MAX_STEPS=-1              # Max steps (-1 = full)
DRIFT_LOG_INTERVAL=100    # Drift tracking frequency
```

### Validation

Before running on real hardware, the implementation was:
- ✅ Designed to reuse existing SNIP computation
- ✅ Verified SNIP score storage uses `args.save` path
- ✅ Configured to use `/dev/shm` for large temporary files
- ✅ Set to save only neuron IDs, not scores (per requirement)
- ✅ Kept clean and simple (fine-tuning ~350 lines)
- ✅ Used standard LoRA via PEFT library

### Dependencies

All dependencies should already be installed (from original repo):
- `torch`
- `transformers`
- `peft` (may need to install: `pip install peft`)
- `datasets`
- `numpy`
- `scipy`
- `matplotlib`

Check with:
```bash
bash scripts/verify_setup.sh
```

### Cleanup After Experiment

Safe to delete (large temporary files):
```bash
rm -rf /dev/shm/snip_scores/
rm -rf /dev/shm/drift_logs/
rm -rf /dev/shm/initial_weights/
```

Keep (small, valuable):
```
neuron_groups/          # Neuron IDs (reproducibility)
finetuned_models/       # Fine-tuned model (future experiments)
results/                # Your findings!
```

### Next Steps

After implementation is ready:
1. Run `bash scripts/verify_setup.sh` to check environment
2. Run full experiment or individual phases
3. Review results in `results/drift_analysis_report.txt`
4. Examine plots in `results/figures/`
5. Interpret findings relative to hypotheses

### Research Contribution

This implementation enables investigation of:
- **Why safety alignment is brittle** during fine-tuning
- **Which neurons change** (safety-critical vs others)
- **How much they change** (quantitative drift metrics)
- **Hypothesis testing** (fragile neurons vs pathway creation)

Complements the original paper's findings on:
- Safety-critical neuron sparsity (~3%)
- Brittleness under pruning
- Ineffectiveness of freezing <50% neurons

### Code Quality

- **Clean**: No modifications to existing code
- **Documented**: Inline comments + extensive docs
- **Modular**: Each phase is independent
- **Configurable**: All parameters in bash scripts
- **Reproducible**: Fixed random seeds, documented steps
- **Efficient**: Uses /dev/shm, LoRA instead of full fine-tuning

## Summary

Implementation complete and ready to run. All code follows requirements:
1. ✅ Potentially large files saved to `/dev/shm`
2. ✅ Only neuron IDs saved, not SNIP scores
3. ✅ LoRA fine-tuning (not full fine-tuning)
4. ✅ Maximal code reuse from original implementation

No execution performed - all code is set up and ready for you to run when desired.
