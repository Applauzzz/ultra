#!/bin/bash
cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "Running pinpoint test (will timeout if hung)..."
timeout 600 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  pinpoint_hang.py 2>&1 | tee pinpoint.log

echo ""
echo "Check pinpoint.log to see where it hung (if timeout occurred)"
tail -30 pinpoint.log
