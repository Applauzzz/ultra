#!/usr/bin/env bash
set -euo pipefail

export MASTER_ADDR=10.60.157.155
export MASTER_PORT=29500
export NNODES=4
export NPROC_PER_NODE=8
export PYTHON=${PYTHON:-python}
export WANDB_OFFICIAL=1;
export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6";
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
HOST=$(hostname)

# conda activate /mnt_upfs/miniconda/envs/ultra
# cd /mnt/zehao/ULTra
# export PYTHONPATH=$(pwd)
# bash /mnt/zehao/ULTra/launch/train_nemo_all.sh
case "$HOST" in
  X2_1|x2-1) export NODE_RANK=0 ;;
  X2_2|x2-2) export NODE_RANK=1 ;;
  X2_3|x2-3) export NODE_RANK=2 ;;
  X2_4|x2-4) export NODE_RANK=3 ;;
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
  /mnt/modelbase/dcp_to_hf.py \
  --dcp_dir /mnt/zehao/modelbase/nemotron/swagdn-8B-20B-3e-5-drop0_5/checkpoints/0000010000\
  --out_dir /mnt/zehao/modelbase/nemotron/hf-swagdn-8B-20B-3e-5-drop0_5/checkpoints/0000010000 \
  --model_name_or_path /mnt/modelbase/modelbase/qwen-swagdn-8b \
  --dtype bf16
# torchrun --standalone --nproc_per_node=8 /mnt/modelbase/dcp_to_hf.py \
#   --dcp_dir /mnt/zehao/modelbase/nemotron/swagdn-8B-10B/checkpoints/0000010000 \
#   --out_dir /mnt/zehao/modelbase/nemotron/swagdn-8B-10B-hf/checkpoints/0000010000 \
#   --model_name_or_path /mnt/modelbase/modelbase/qwen-swagdn-8b \
#   --dtype fp32 

# bash /mnt/zehao/ULTra/launch/load_dcp.sh