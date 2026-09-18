#!/bin/bash
cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "Testing if WandB causes worker deadlock..."
timeout 180 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  test_wandb_fork.py 2>&1 | tee wandb_fork_test.log

echo ""
tail -30 wandb_fork_test.log
