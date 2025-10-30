# Quick Start: PEFT Fine-Tuning

## One-Command Test

```bash
# Verify installation
python verify_peft_installation.py
```

## Three-Step Experiment

### 1. Identify Safety Neurons
```bash
python main_extension.py \
    --task identify_safety_neurons \
    --model llama2-7b-chat-hf \
    --sparsity_ratio 0.03 \
    --results_path ./results/peft
```

### 2. Fine-Tune with PEFT
```bash
python main_extension.py \
    --task fine_tune_frozen \
    --model llama2-7b-chat-hf \
    --safety_masks ./results/peft/safety_neurons/original_safety_masks.pt \
    --use_peft \
    --results_path ./results/peft
```

### 3. Analyze & Plot
```bash
# Compute statistics
python main_extension.py \
    --task eval_score_dynamics \
    --original_safety_scores_path ./results/peft/safety_neurons/original_safety_scores.pt \
    --fine_tuned_safety_scores_path ./results/peft/safety_neurons/fine_tuned_safety_scores.pt \
    --original_safety_masks_path ./results/peft/safety_neurons/original_safety_masks.pt \
    --results_path ./results/peft

# Generate plots
python plot_wanda_scores.py \
    --original_scores ./results/peft/safety_neurons/original_safety_scores.pt \
    --fine_tuned_scores ./results/peft/safety_neurons/fine_tuned_safety_scores.pt \
    --masks ./results/peft/safety_neurons/original_safety_masks.pt \
    --output_dir ./results/peft/plots
```

## Key Parameters

**LoRA Configuration:**
- `--use_peft`: Enable LoRA
- `--peft_r 8`: Rank (8=efficient, 16=more capacity)
- `--peft_alpha 16`: Alpha scaling (typically 2x rank)
- `--peft_target_modules q_proj,v_proj`: Which layers to adapt

**Experiment Configuration:**
- `--sparsity_ratio 0.03`: % of neurons to identify as safety-critical (3% from paper)
- `--num_epochs 3`: Training epochs
- `--nsamples 1000`: Training examples (default in `get_loaders()`)

## Expected Output

**Step 1 Output:**
```
Identified safety-critical neurons in 32 layers
Total safety-critical neurons: 50000/16700000 (3.00%)
Saved to: ./results/peft/safety_neurons/
```

**Step 2 Output:**
```
Applying PEFT (LoRA) with r=8, alpha=16
trainable params: 20971520 || all params: 6738415616 || trainable%: 0.31%
Freezing safety-critical neurons...
Frozen 50000/16700000 safety-critical neurons (3.00%)
Note: LoRA adapters remain trainable (only base model neurons frozen)
Fine-tuning completed. Model saved to /dev/shm/fine_tuned_model
```

**Step 3 Output:**
```
SCORE DYNAMICS SUMMARY
Original Safety Neurons:
  Mean: 0.123456, Std: 0.045678, Median: 0.098765
Fine-tuned Safety Neurons:
  Mean: 0.109876, Std: 0.043210, Median: 0.087654
Overall Score Change: -11.02%
→ Supports Hypothesis A: Representational drift causing score drops
```

**Plots Generated:**
- `wanda_score_distributions.png`: Histogram comparison
- `wanda_score_statistics.png`: Bar chart of metrics

## Interpretation

**Score Drop > 10%:**
- Hypothesis A (Representational Drift)
- LoRA adapters successfully bypass frozen neurons
- Safety alignment is brittle to low-rank modifications

**Score Stable < 5%:**
- Hypothesis B (Global Redistribution)
- Frozen neurons maintain importance
- Safety degradation from context shifts, not local drift

## Troubleshooting

**"No module named 'peft'":**
```bash
pip install peft
```

**"No module named 'matplotlib'":**
```bash
pip install matplotlib numpy
```

**Out of memory:**
- Reduce `--batch_size` (default: 4)
- Reduce `--peft_r` (try r=4)
- Enable more aggressive gradient checkpointing

**PEFT not applying:**
- Ensure `--use_peft` flag is set (it's a boolean flag, not a value)
- Check logs for "Applying PEFT (LoRA)" message

## Documentation

- Full guide: [PEFT_README.md](PEFT_README.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- Extension overview: [EXTENSION_DOCUMENTATION.md](EXTENSION_DOCUMENTATION.md)
- Main README: [README.md](README.md)
