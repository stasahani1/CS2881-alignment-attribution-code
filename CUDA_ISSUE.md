# CUDA Initialization Issue

## Problem
PyTorch cannot initialize CUDA in this RunPod container environment. Any operation that tries to allocate GPU memory fails with:

```
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

## Diagnostics Performed
- ✅ GPU is visible and functional (`nvidia-smi` works)
- ✅ No processes using GPU
- ✅ PyTorch detects CUDA (`torch.cuda.is_available() == True`)
- ✅ CUDA libraries are present and linked correctly
- ✅ Container restarted
- ✅ No IPC/semaphore conflicts
- ✅ Driver version compatible (575.57.08, supports CUDA 12.9)
- ✅ PyTorch compiled with CUDA 12.1 (compatible)
- ❌ **Any CUDA memory allocation fails**

## Root Cause
This is a PyTorch 2.1.0+cu121 + RunPod container interaction issue. The specific error occurs when:
1. PyTorch tries to initialize CUDA context
2. Specifically in `accelerate` library's `get_max_memory()` function
3. When calling `torch.tensor([0], device='cuda')`

## Workarounds Attempted

### 1. Changed `device_map="auto"` to manual `.to('cuda')`
**Status**: Implemented in all `main*.py` files
**Result**: Still fails because the underlying `model.to('cuda')` also tries to allocate GPU memory

### 2. Different temp directories
**Status**: Tried `/workspace/tmp`, `/dev/shm`
**Result**: No change

### 3. Environment variables
**Status**: Tried various CUDA environment settings
**Result**: No change

## Possible Solutions

### Option 1: Use CPU-only mode (Testing)
For testing the pipeline without GPU:
```bash
export CUDA_VISIBLE_DEVICES=""
```
This will force CPU mode, but will be very slow for the 7B model.

### Option 2: Different RunPod Template
The issue might be specific to this RunPod template/image. Try:
1. Create a new pod with a different PyTorch template
2. Or use RunPod's official PyTorch 2.1 template

### Option 3: Rebuild PyTorch
Reinstall PyTorch with matching CUDA version:
```bash
pip3 install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```
Warning: This might break other dependencies.

### Option 4: Contact RunPod Support
This appears to be a container/driver configuration issue that RunPod support might need to address.

## Code Changes Made

All model loading functions updated to avoid `device_map="auto"`:

###Before:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Uses accelerate
    ...
)
```

### After:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # No device_map
    ...
)
model = model.to('cuda:0')  # Manual move
```

**Files modified**:
- `main.py`
- `main_low_rank.py`
- `main_low_rank_diff.py`

## Next Steps

1. **Try a fresh RunPod pod** with PyTorch 2.1 or 2.5 template
2. **Test with a minimal CUDA program** to isolate if it's PyTorch-specific
3. **Check RunPod community** forums for similar issues
4. **Contact RunPod support** with this diagnostic information

## Testing Command
To verify if CUDA works:
```bash
source .venv/bin/activate
python3 -c "import torch; x = torch.zeros(1).cuda(); print('CUDA works!')"
```

If this works, the experiments should run.
