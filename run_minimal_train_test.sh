#!/bin/bash
set -e

source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
unset NCCL_IB_DISABLE
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/zehao/ultra
echo "Running minimal training test..."
timeout 180 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 test_minimal_train.py
