export WANDB_OFFICIAL=1;
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HF_DATASETS_CACHE="/home/zhliu/database/hf-cache";
# export HUGGINGFACE_HUB_CACHE="/home/zhliu/database/hub_cache";
export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6";
# export PYTHONPATH=$(pwd)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
# ===== 强制走 IB/RDMA，先固定一组口 =====
unset NCCL_IB_DISABLE
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torchrun --nproc_per_node=1 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=10.60.137.131 \
#         --master_port=29501 \
#         ./main/train_hgrn_ori.py config=./main/model_config/hgrn/hgrn.yaml\
#         2>&1 | tee /mnt/zehao/output_hgrn.log


# set -euo pipefail

export WANDB_OFFICIAL=1
export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0

BASE_CONFIG=/mnt/zehao/ULTra/main/configs/act_fsdp2_sp4/gla_7B_1B.yaml
TMP_CONFIG_DIR=/mnt/zehao/ULTra/main/model_config/tmp_configs
LOG_DIR=/mnt/zehao/ULTra/main/model_log/log
CKPT_ROOT=/mnt/zehao/ULTra/main/model_ckpt/ckpt

mkdir -p "$TMP_CONFIG_DIR" "$LOG_DIR"

# ===== 起点：1M =====
# seq_len=128
# seq_len=1048576   # 1M
seq_len=262144 #256k
#512k
# seq_len=524288
# max_seq_len=524288
max_seq_len=268435456  # 256M
NNODES=1
NPROC_PER_NODE=8
MASTER_ADDR=10.60.157.155
MASTER_PORT=29502

LOCAL_IP=$(hostname -I | awk '{print $1}')

case "$LOCAL_IP" in
  10.60.157.155)
    NODE_RANK=0
    ;;
  10.60.35.33)
    NODE_RANK=1
    ;;
  10.60.233.146)
    NODE_RANK=0
    ;;
  10.60.137.131)
    NODE_RANK=0
    ;;
  *)
    echo "Unknown machine IP: $LOCAL_IP"
    exit 1
    ;;
esac


torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node=8 \
    --node_rank="$NODE_RANK" \
    --master_addr="$LOCAL_IP" \
    --master_port="$MASTER_PORT" \
    ./main/train_freelong_1b.py \
    config="/mnt/zehao/ULTra/main/configs/rebuttal/gla_1.3B.yaml" 
    # 2>&1 | tee "$log_file"


