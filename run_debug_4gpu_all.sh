#!/bin/bash
set -e
cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "Debugging 4 GPU + all subset DataLoader..."
timeout 300 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  debug_4gpu_all.py 2>&1 | tee debug_4gpu_all.log

echo ""
echo "Check debug_4gpu_all.log for results"
