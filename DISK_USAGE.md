# Disk Usage Strategy for H100 RunPod

## Storage Partitions

Your RunPod container has three storage areas:

| Partition | Size | Purpose | Current Usage |
|-----------|------|---------|---------------|
| `/workspace` | 50GB | Code, results, wanda scores | ~7GB (.venv + data) |
| `/tmp` (overlay) | 30GB | System temp files only | ~600MB |
| `/dev/shm` | 117GB | **PRIMARY STORAGE** - models, cache, checkpoints | ~26GB (HF cache) |

## File Storage Locations

### HuggingFace Models (Downloaded)
- **Location**: `/dev/shm/models/llama-2-7b-chat-hf/`
- **Size**: ~13GB
- **Why /dev/shm**: Large storage (117GB) prevents disk full errors

### HuggingFace Cache
- **Location**: `/dev/shm/huggingface/` (symlinked from `/tmp/huggingface/`)
- **Size**: ~26GB (includes model metadata, blobs, and refs)
- **Environment Variables**:
  - `HF_HOME=/dev/shm/huggingface`
  - `HF_HUB_CACHE=/dev/shm/huggingface/hub`
- **Why /dev/shm**: The HF cache can grow very large (26GB+) and was filling /tmp, causing disk full errors

### Fine-tuned Models (Experiments 1 & 2)
- **Location**: `/dev/shm/fine_tuned_models/`
- **Size**: ~13GB per fine-tuned model
- **Default Path**: `/dev/shm/fine_tuned_model`
- **Script Paths**:
  - Experiment 1: `/dev/shm/fine_tuned_models/llama2-7b-chat-hf/experiment_1_frozen`
  - Experiment 2: `/dev/shm/fine_tuned_models/llama2-7b-chat-hf/experiment_2_unfrozen`
- **Why /dev/shm**: Largest partition (117GB), fast shared memory

### Pruned Models (from main.py)
- **Location**: `/dev/shm/pruned_models/`
- **Why /dev/shm**: Large storage for temporary pruned models

### Checkpoint Files (rewind_ft_model.py)
- **Location**: `/dev/shm/ckpts/`
- **Why /dev/shm**: Large storage for temporary checkpoints

### Results and Analysis
- **Location**: `/workspace/results/`
- **Contents**:
  - Wanda scores (top k% only - memory efficient)
  - Safety neuron masks
  - Analysis results
  - Attack evaluation results
- **Why /workspace**: Persistent storage for important results

### Output Directories (from bash scripts)
- **Location**: `/workspace/out/`
- **Contents**: Wanda scores, masks, pruning results
- **Why /workspace**: Small persistent results

## Disk Usage by Operation

### 1. Downloading Models (`download_hf_models.py`)
```bash
uv run python download_hf_models.py
```
- Downloads to: `/dev/shm/models/llama-2-7b-chat-hf/`
- Cache: `/dev/shm/huggingface/`
- Disk usage: ~13GB model + ~26GB cache = ~39GB in `/dev/shm`

### 2. Running Experiments (Extension)
```bash
# Experiment 1 (Frozen)
./scripts/01_run_experiment_1_frozen.sh
# - Fine-tuned model: /dev/shm/fine_tuned_models/*/experiment_1_frozen (~13GB)
# - Results: /workspace/results/ (<1GB)

# Experiment 2 (Unfrozen)
./scripts/02_run_experiment_2_unfrozen.sh
# - Fine-tuned model: /dev/shm/fine_tuned_models/*/experiment_2_unfrozen (~13GB)
# - Results: /workspace/results/ (<1GB)
```

### 3. Pruning (main.py)
```bash
python main.py --model llama2-7b-chat-hf --prune_method wanda ...
```
- Loads from: `/dev/shm/models/llama-2-7b-chat-hf/`
- Saves to: `/workspace/out/` (wanda scores)
- Temp models: `/dev/shm/pruned_models/`

## Storage Optimization

### Current Optimizations
1. **Top k% Wanda Scores Only**: Only saving top k% of safety neuron scores (your recent change)
2. **Checkpoint Limits**: `save_total_limit=2` keeps only last 2 checkpoints during training
3. **Model Paths**: All main scripts updated to use `/tmp` or `/dev/shm`

### Manual Cleanup Commands
```bash
# Clean HuggingFace cache (frees ~26GB in /dev/shm)
rm -rf /dev/shm/huggingface/*

# Clean downloaded models (frees ~13GB in /dev/shm)
rm -rf /dev/shm/models/*

# Clean fine-tuned models (frees ~26GB in /dev/shm)
rm -rf /dev/shm/fine_tuned_models/*

# Clean pruned models (variable size in /dev/shm)
rm -rf /dev/shm/pruned_models/*

# Clean temporary checkpoints (frees variable in /dev/shm)
rm -rf /dev/shm/ckpts/*

# Complete /dev/shm cleanup (WARNING: removes everything!)
rm -rf /dev/shm/*

# Use the provided cleanup script
./scripts/cleanup_tmp.sh
```

### Check Disk Usage
```bash
# Overall usage
df -h /workspace / /dev/shm

# Detailed directory sizes
du -sh /workspace/results /workspace/out /dev/shm/*

# Check what's using space
du -sh /dev/shm/* | sort -h
```

## Troubleshooting

### "No space left on device" errors

1. **If /workspace is full (50GB)**:
   - Check: `du -sh /workspace/results /workspace/out /workspace/.venv`
   - Solution: Only keep essential results, don't store models here

2. **If / (overlay/tmp) is full (30GB)**:
   - Check: `df -h /` and `du -sh /tmp/*`
   - Solution: This should stay < 1GB. If full, check for stray files in /tmp
   - **IMPORTANT**: HuggingFace cache is now in `/dev/shm`, NOT `/tmp`

3. **If /dev/shm is full (117GB)**:
   - Check: `du -sh /dev/shm/* | sort -h`
   - Most likely culprits:
     - HuggingFace cache: ~26GB (`/dev/shm/huggingface/`)
     - Downloaded model: ~13GB (`/dev/shm/models/`)
     - Fine-tuned models: ~13-26GB each (`/dev/shm/fine_tuned_models/`)
     - Pruned models: variable (`/dev/shm/pruned_models/`)
   - Solution: Clear old fine-tuned models with `rm -rf /dev/shm/fine_tuned_models/*`

### Model Loading Issues

If model loading fails with "FileNotFoundError":
1. Ensure model is downloaded: `ls -lh /dev/shm/models/llama-2-7b-chat-hf/`
2. Re-download if needed: `uv run python download_hf_models.py`
3. Check HF authentication: `huggingface-cli login`

### CUDA Errors

If you get "CUDA error: CUDA-capable device(s) is/are busy or unavailable":
1. Check disk space: `df -h /` - If overlay is full, CUDA can't create temp files
2. Solution: Free up space by moving HF cache to `/dev/shm` (already done)
3. Verify: `nvidia-smi` should show no errors

## Files Modified for Disk Optimization

All files now use `/dev/shm` (117GB) instead of `/tmp` (30GB) to prevent disk full errors:

1. `download_hf_models.py` - Download to `/dev/shm/models/`, cache to `/dev/shm/huggingface/`
2. `main.py` - HF_HOME to `/dev/shm/huggingface/`, model path to `/dev/shm/models/`, saves to `/dev/shm/pruned_models/`
3. `main_extension.py` - HF_HOME to `/dev/shm/huggingface/`, default save to `/dev/shm/fine_tuned_model`
4. `main_low_rank.py` - HF_HOME to `/dev/shm/huggingface/`, model path to `/dev/shm/models/`, saves to `/dev/shm/pruned_models/`
5. `main_low_rank_diff.py` - HF_HOME to `/dev/shm/huggingface/`, model path to `/dev/shm/models/`
6. `lib/extension_utils.py` - Default save to `/dev/shm/fine_tuned_model`
7. `rewind_ft_model.py` - HF_HOME to `/dev/shm/huggingface/`, checkpoints to `/dev/shm/ckpts/`
8. `scripts/01_run_experiment_1_frozen.sh` - Models to `/dev/shm/`
9. `scripts/02_run_experiment_2_unfrozen.sh` - Models to `/dev/shm/`

## Important: Symlink Compatibility

For backwards compatibility, `/tmp/huggingface` is symlinked to `/dev/shm/huggingface/`:
```bash
/tmp/huggingface -> /dev/shm/huggingface/
```
This ensures any code that still references `/tmp/huggingface` will work correctly.
