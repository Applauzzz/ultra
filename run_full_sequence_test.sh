#!/bin/bash
cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "Replicating EXACT training initialization sequence..."
echo "This includes: model, DDP, optimizer, checkpoint, WandB, gc.disable"
echo ""

timeout 240 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  test_full_init_sequence.py 2>&1 | tee full_sequence_test.log

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Full sequence completed successfully"
    echo "The deadlock is NOT reproducible with this sequence"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "✗ TIMEOUT - Deadlock reproduced!"
    echo "Last step completed:"
    grep "✓ STEP" full_sequence_test.log | tail -1
else
    echo "✗ Failed with exit code $EXIT_CODE"
fi
echo "=========================================="

tail -30 full_sequence_test.log
