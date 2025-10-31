#!/bin/bash
# Verify that the experimental setup is ready to run

echo "========================================================================"
echo "SETUP VERIFICATION"
echo "========================================================================"
echo ""

# Check Python packages
echo "Checking Python packages..."
REQUIRED_PACKAGES=(
    "torch"
    "transformers"
    "peft"
    "datasets"
    "numpy"
    "scipy"
    "matplotlib"
)

MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    python -c "import $pkg" 2>/dev/null
    if [ $? -ne 0 ]; then
        MISSING_PACKAGES+=("$pkg")
        echo "  ✗ $pkg - NOT FOUND"
    else
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        echo "  ✓ $pkg - $VERSION"
    fi
done

echo ""

# Check disk space
echo "Checking disk space..."
echo "  /dev/shm:"
df -h /dev/shm | tail -1 | awk '{print "    Total: "$2", Available: "$4", Used: "$5}'

echo "  /workspace:"
df -h /workspace | tail -1 | awk '{print "    Total: "$2", Available: "$4", Used: "$5}'

echo "  /tmp:"
df -h /tmp | tail -1 | awk '{print "    Total: "$2", Available: "$4", Used: "$5}'

echo ""

# Check GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "  ✓ Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | while read line; do
        echo "    $line"
    done
else
    echo "  ✗ nvidia-smi not found"
fi

echo ""

# Check model path
echo "Checking model path..."
MODEL_PATH="/tmp/llama-2-7b-chat-hf/"
if [ -d "$MODEL_PATH" ]; then
    echo "  ✓ Model found at $MODEL_PATH"
    if [ -f "$MODEL_PATH/config.json" ]; then
        echo "    ✓ config.json exists"
    else
        echo "    ✗ config.json missing"
    fi
else
    echo "  ✗ Model not found at $MODEL_PATH"
    echo "    Please download the model or update MODEL_PATH in scripts"
fi

echo ""

# Check data files
echo "Checking data files..."
DATA_FILES=(
    "data/SFT_aligned_llama2-7b-chat-hf_train.csv"
    "data/advbench.txt"
)

for file in "${DATA_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ $file - NOT FOUND"
    fi
done

echo ""

# Check scripts are executable
echo "Checking scripts..."
SCRIPTS=(
    "scripts/phase1_compute_snip_scores.sh"
    "scripts/phase1_identify_neurons.sh"
    "scripts/phase2_finetune.sh"
    "scripts/run_full_experiment.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo "  ✓ $script (executable)"
    elif [ -f "$script" ]; then
        echo "  ⚠ $script (not executable, run: chmod +x $script)"
    else
        echo "  ✗ $script - NOT FOUND"
    fi
done

echo ""

# Summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo "✓ All required packages are installed"
else
    echo "✗ Missing packages: ${MISSING_PACKAGES[*]}"
    echo "  Install with: pip install ${MISSING_PACKAGES[*]}"
fi

echo ""
echo "Estimated disk space required:"
echo "  - SNIP scores: ~5-10 GB (/dev/shm)"
echo "  - Initial weights: ~1-2 GB (/dev/shm)"
echo "  - Drift logs: ~2-5 GB (/dev/shm)"
echo "  - Fine-tuned model: ~7 GB (/workspace)"
echo "  - Total: ~15-25 GB"
echo ""

# Check if sufficient space
DEVSHM_AVAIL=$(df /dev/shm | tail -1 | awk '{print $4}')
WORKSPACE_AVAIL=$(df /workspace | tail -1 | awk '{print $4}')

if [ $DEVSHM_AVAIL -lt 10485760 ]; then  # Less than 10GB in KB
    echo "⚠ Warning: /dev/shm has less than 10GB available"
fi

if [ $WORKSPACE_AVAIL -lt 10485760 ]; then  # Less than 10GB in KB
    echo "⚠ Warning: /workspace has less than 10GB available"
fi

echo ""
echo "Ready to run? Use one of:"
echo "  bash scripts/run_full_experiment.sh        (Full pipeline)"
echo "  bash scripts/phase1_compute_snip_scores.sh (Phase 1a only)"
echo ""
