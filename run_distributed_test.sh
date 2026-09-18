#!/bin/bash
set -e

echo "Running distributed DataLoader test (4 GPUs)"
echo "============================================="

export CUDA_VISIBLE_DEVICES=4,5,6,7

cd /mnt/zehao/ultra

# Run with torchrun
echo "Starting test with torchrun..."
timeout 180 torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  test_distributed_dataloader.py || {
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo ""
        echo "ERROR: Test timed out after 180 seconds"
        echo "DataLoader is hanging in distributed mode"
    else
        echo ""
        echo "ERROR: Test failed with exit code $exit_code"
    fi
    exit $exit_code
}

echo ""
echo "============================================="
echo "Distributed test completed successfully!"
echo "============================================="
