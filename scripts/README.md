# Extension Scripts

Automated bash scripts for running safety-critical neuron analysis experiments.

## Quick Start

### 0. Check Status (Optional)
```bash
./scripts/check_status.sh
```
See which experiments have already been completed.

### 1. Quick Test (Recommended First Step)
```bash
./scripts/quick_test.sh
```
Runs the full pipeline with minimal settings (~30-60 min) to verify everything works.

### 2. Full Pipeline (Single Model)
```bash
./scripts/run_full_pipeline.sh
```
Runs all experiments with default settings on llama2-7b-chat-hf.

### 3. Custom Configuration
```bash
# Copy and edit the config file
cp scripts/config_example.sh scripts/my_config.sh
nano scripts/my_config.sh

# Run with custom settings
source scripts/my_config.sh
./scripts/run_full_pipeline.sh
```

### 4. Multi-Model Comparison
```bash
MODELS="llama2-7b-chat-hf llama2-13b-chat-hf" ./scripts/compare_models.sh
```

## Script Overview

### Core Scripts

| Script | Description | Runtime |
|--------|-------------|---------|
| `00_setup_neuron_identification.sh` | Identify safety/utility neurons | ~2-4 hours |
| `01_run_experiment_1_frozen.sh` | Frozen-regime fine-tuning | ~4-8 hours |
| `02_run_experiment_2_unfrozen.sh` | Unfrozen fine-tuning | ~4-8 hours |

### Utility Scripts

| Script | Description |
|--------|-------------|
| `run_full_pipeline.sh` | Run all steps sequentially |
| `quick_test.sh` | Fast test with minimal settings |
| `compare_models.sh` | Run experiments on multiple models |
| `check_status.sh` | Check which experiments are complete |
| `config_example.sh` | Template for configuration |

## Configuration Variables

All scripts accept these environment variables:

### Model Settings
- `MODEL` - Model name (default: `llama2-7b-chat-hf`)
- `DEVICE` - GPU device (default: `cuda:0`)

### Neuron Identification
- `PRUNE_METHOD` - Method for scoring (default: `wanda`)
  - `wanda`: Magnitude-based (|w| × √activation)
  - `wandg`: Gradient-based (SNIP)
- `SPARSITY_RATIO` - Fraction of neurons to identify (default: `0.05`)
  - Recommended: 0.01-0.1
- `NSAMPLES` - Number of samples for scoring (default: `128`)

### Fine-tuning
- `TRAINING_DATA` - Dataset for fine-tuning (default: `alpaca_cleaned_no_safety`)
- `NUM_EPOCHS` - Number of epochs (default: `3`)
- `LEARNING_RATE` - Learning rate (default: `2e-5`)
- `BATCH_SIZE` - Batch size (default: `4`)
- `MAX_LENGTH` - Max sequence length (default: `512`)

### Evaluation
- `EVAL_ATTACK` - Run ASR evaluation (default: `false`)
  - Set to `true` to evaluate attack success rate (requires vLLM)

### Other
- `SEED` - Random seed (default: `42`)

## Examples

### Example 1: Quick Test
```bash
./scripts/quick_test.sh
```

### Example 2: Full Run with Custom Settings
```bash
MODEL=llama2-7b-chat-hf \
SPARSITY_RATIO=0.05 \
NUM_EPOCHS=3 \
EVAL_ATTACK=true \
./scripts/run_full_pipeline.sh
```

### Example 3: Only Run Neuron Identification
```bash
MODEL=llama2-13b-chat-hf \
SPARSITY_RATIO=0.1 \
./scripts/00_setup_neuron_identification.sh
```

### Example 4: Run Experiment 1 Only (Assumes neurons already identified)
```bash
MODEL=llama2-7b-chat-hf \
NUM_EPOCHS=5 \
./scripts/01_run_experiment_1_frozen.sh
```

### Example 5: Compare Multiple Models
```bash
MODELS="llama2-7b-chat-hf llama2-13b-chat-hf" \
SPARSITY_RATIO=0.05 \
NUM_EPOCHS=3 \
EVAL_ATTACK=true \
./scripts/compare_models.sh
```

### Example 6: Use Config File
```bash
# Edit config
cp scripts/config_example.sh scripts/my_experiment.sh
nano scripts/my_experiment.sh

# Run with config
source scripts/my_experiment.sh
./scripts/run_full_pipeline.sh
```

## Output Structure

```
results/
├── llama2-7b-chat-hf/
│   ├── neuron_identification/
│   │   ├── safety_neurons/
│   │   │   ├── original_safety_masks.pt
│   │   │   └── original_safety_scores.pt
│   │   └── utility_neurons/
│   │       ├── original_utility_masks.pt
│   │       └── original_utility_scores.pt
│   │
│   ├── experiment_1_frozen/
│   │   ├── fine_tuned_model/
│   │   ├── safety_neurons/
│   │   │   ├── fine_tuned_safety_masks.pt
│   │   │   └── fine_tuned_safety_scores.pt
│   │   ├── score_dynamics_analysis.json
│   │   ├── asr_results_frozen.json (if EVAL_ATTACK=true)
│   │   └── attack_results/ (if EVAL_ATTACK=true)
│   │
│   └── experiment_2_unfrozen/
│       ├── fine_tuned_model/
│       ├── original_weights.pt
│       ├── weight_drift_analysis.json
│       ├── asr_results_unfrozen.json (if EVAL_ATTACK=true)
│       └── attack_results/ (if EVAL_ATTACK=true)
│
└── llama2-13b-chat-hf/
    └── ... (same structure)
```

## Key Results Files

### Experiment 1 (Frozen-Regime)
- **`score_dynamics_analysis.json`**
  - Check `score_distributions.original_safety.mean` vs `new_safety.mean`
  - Large drop (-10%+) → Hypothesis A (Representational Drift)
  - Stable (near 0%) → Hypothesis B (Global Redistribution)

### Experiment 2 (Unfrozen)
- **`weight_drift_analysis.json`**
  - Check `safety.cosine_similarity.mean` vs `random.cosine_similarity.mean`
  - Safety < Random → Hypothesis C (Fragile Safety Neurons)
  - Safety ≈ Random → Hypothesis D (Recontextualization)

### ASR Results (if EVAL_ATTACK=true)
- **`asr_results_frozen.json`** / **`asr_results_unfrozen.json`**
  - Compare `average_asr` across experiments
  - Higher ASR = more unsafe model

## GPU Memory Requirements

**Your GPU: 44GB** → Use the optimized config:
```bash
source scripts/config_44gb_gpu.sh
./scripts/run_full_pipeline.sh
```

See [GPU_REQUIREMENTS.md](../GPU_REQUIREMENTS.md) for detailed memory analysis.

## Troubleshooting

### Out of Memory
The code is already optimized for 44GB GPUs. If you still get OOM:

```bash
# Further reduce memory usage
BATCH_SIZE=1 \
MAX_LENGTH=128 \
NSAMPLES=64 \
./scripts/run_full_pipeline.sh
```

### Prerequisites Not Met
If you see "ERROR: Safety masks not found", run:
```bash
./scripts/00_setup_neuron_identification.sh
```

### Verify Setup
Check that required files exist:
```bash
ls results/${MODEL}/neuron_identification/safety_neurons/
ls results/${MODEL}/neuron_identification/utility_neurons/
```

## Time Estimates

**Full Pipeline (default settings on A100 GPU):**
- Neuron identification: ~2-4 hours
- Experiment 1 (frozen): ~4-8 hours
- Experiment 2 (unfrozen): ~4-8 hours
- **Total: ~10-20 hours**

**Quick Test (minimal settings):**
- ~30-60 minutes

**Multi-Model Comparison (2 models):**
- ~20-40 hours (2× single model time)

## Tips

1. **Always run `quick_test.sh` first** to catch issues early
2. **Use `screen` or `tmux`** for long-running experiments
3. **Monitor GPU usage** with `nvidia-smi`
4. **Check disk space** - fine-tuned models are ~13GB each
5. **Set `EVAL_ATTACK=false`** if you don't need ASR (saves time)
6. **Use lower settings** for debugging:
   ```bash
   SPARSITY_RATIO=0.01 NSAMPLES=32 NUM_EPOCHS=1 BATCH_SIZE=2
   ```

## Next Steps

After experiments complete:
1. Analyze `score_dynamics_analysis.json` (Experiment 1)
2. Analyze `weight_drift_analysis.json` (Experiment 2)
3. Compare ASR results if `EVAL_ATTACK=true`
4. Create visualizations of score distributions and drift metrics
5. Write up findings based on hypothesis testing
