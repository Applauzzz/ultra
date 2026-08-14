#!/usr/bin/env bash
set -euo pipefail

export MASTER_ADDR=10.60.233.146 
export MASTER_PORT=29502
export NNODES=2
export NPROC_PER_NODE=8
export PYTHON=${PYTHON:-python}
export WANDB_OFFICIAL=1;
export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6";
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
HOST=$(hostname)

# conda activate /mnt_upfs/miniconda/envs/ultra_2
# cd /mnt/zehao/ULTra
export PYTHONPATH=$(pwd)
# bash /mnt/zehao/ULTra/launch/train_nemo_all.sh
# bash /mnt/zehao/ULTra/launch/train_nemo_all.sh > /mnt/zehao/ULTra/logs/train_${HOSTNAME}.log 2>&1
case "$HOST" in
  X2_1|x2-1) export NODE_RANK=0 ;;
  X2_2|x2-2) export NODE_RANK=1 ;;
  X2_3|x2-3) export NODE_RANK=0 ;;
  X2_4|x2-4) export NODE_RANK=1 ;;
  *)
    echo "Unknown hostname: $HOST"
    exit 1
    ;;
esac

# ===== NCCL / torch debug =====
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET,COLL,GRAPH
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_ASYNC_ERROR_HANDLING=1

# ===== 强制走 IB/RDMA，先固定一组口 =====
unset NCCL_IB_DISABLE
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1

# 可选：减少杂音
export OMP_NUM_THREADS=1

echo "HOST=$HOST"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NCCL_IB_HCA=$NCCL_IB_HCA"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"

torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ./main/train.py config=/mnt/zehao/ULTra/main/model_config/nemo/swagdn_1B_32k.yaml