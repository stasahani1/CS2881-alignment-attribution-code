#!/bin/bash
################################################################################
# Script: cleanup_tmp.sh
# Description: Cleans up large model checkpoints from /tmp/ to free space
################################################################################

set -e

TMP_MODELS_DIR="/tmp/fine_tuned_models"

echo "======================================================================"
echo "CLEANUP: Temporary Model Checkpoints"
echo "======================================================================"

if [ ! -d "${TMP_MODELS_DIR}" ]; then
    echo "No temporary models found at ${TMP_MODELS_DIR}"
    echo "Nothing to clean up."
    exit 0
fi

# Show current disk usage
echo "Current /tmp/ usage:"
du -sh ${TMP_MODELS_DIR}/* 2>/dev/null || echo "  (empty)"
echo ""

# Confirm before deleting
read -p "Delete all model checkpoints in ${TMP_MODELS_DIR}? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting ${TMP_MODELS_DIR}..."
    rm -rf ${TMP_MODELS_DIR}
    echo "✓ Cleanup complete"
else
    echo "Cleanup cancelled"
fi

echo "======================================================================"
