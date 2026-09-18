#!/bin/bash
set -e

echo "Running standalone DataLoader test"
echo "==================================="

cd /mnt/zehao/ultra

# Activate environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
else
    echo "Activating eeg-infra environment..."
    source /mnt_upfs/miniconda/etc/profile.d/conda.sh
    conda activate eeg-infra
fi

echo ""
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run test with timeout
timeout 300 python test_dataloader_only.py || {
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo ""
        echo "ERROR: Test timed out after 300 seconds (5 minutes)"
        echo "This suggests the DataLoader is hanging during initialization or iteration"
    else
        echo ""
        echo "ERROR: Test failed with exit code $exit_code"
    fi
    exit $exit_code
}

echo ""
echo "==================================="
echo "Test completed successfully!"
echo "==================================="
