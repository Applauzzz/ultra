#!/bin/bash
# Test if NCCL environment variables cause worker deadlock

cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

echo "TEST 1: WITHOUT NCCL environment variables"
echo "==========================================="

export CUDA_VISIBLE_DEVICES=4,5,6,7
unset NCCL_IB_HCA
unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME

timeout 120 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  test_distributed_dataloader.py 2>&1 | grep -E "Iterator created|Batch.*loaded|SUCCESS|TIMEOUT" | head -10

echo ""
echo "TEST 2: WITH NCCL environment variables (like training)"
echo "==========================================="

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1

timeout 120 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  test_distributed_dataloader.py 2>&1 | grep -E "Iterator created|Batch.*loaded|SUCCESS|TIMEOUT" | head -10

echo ""
echo "If TEST 2 is slower or hangs, NCCL env vars cause the issue!"
