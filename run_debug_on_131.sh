#!/bin/bash
# Run debug training on 10.60.137.131

echo "Running debug training on 10.60.137.131..."
echo "This will identify exactly where the training hangs."
echo ""

ssh 10.60.137.131 'bash -s' << 'REMOTE'
set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/zehao/ultra

source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

echo "Starting debug test..."
echo "Output will be written to debug_train.log"
echo ""

# Run with timeout to prevent indefinite hang
timeout 300 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  debug_train.py 2>&1 | tee debug_train.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Debug test PASSED!"
    echo "All steps completed successfully."
elif [ $EXIT_CODE -eq 124 ]; then
    echo "✗ Debug test TIMED OUT (5 minutes)"
    echo "Last completed step:"
    grep "✓ STEP" debug_train.log | tail -1
    echo ""
    echo "Hang location:"
    tail -5 debug_train.log
else
    echo "✗ Debug test FAILED with exit code $EXIT_CODE"
    echo "Last output:"
    tail -20 debug_train.log
fi
echo "=========================================="
REMOTE
