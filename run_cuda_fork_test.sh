#!/bin/bash
cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "Testing if CUDA + barrier before iter() causes hang..."
timeout 180 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  test_cuda_fork_issue.py 2>&1 | tee cuda_fork_test.log

echo ""
echo "Results in cuda_fork_test.log"
echo "If TEST 2 hangs, that confirms CUDA context pollution in workers."
