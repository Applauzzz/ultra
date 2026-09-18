#!/bin/bash
# Test EEG training on 4 GPUs (GPU 4,5,6,7)

set -e

echo "========================================"
echo "Testing EEG Training on 4 GPUs (4-7)"
echo "========================================"

# Activate conda environment
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
# unset NCCL_IB_DISABLE
# export NCCL_IB_HCA=mlx5_0
# export NCCL_SOCKET_IFNAME=eth1
# export GLOO_SOCKET_IFNAME=eth1
# Use GPUs 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration
CONFIG="configs/eeg/base.yaml"
STEPS=10000
NAME="eeg_test_4gpu_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  Python: $(which python)"
echo "  GPUs: 4,5,6,7"
echo "  Config: $CONFIG"
echo "  Steps: $STEPS"
echo "  Name: $NAME"
echo ""

# Run training with 4 GPUs (DDP mode)
echo "Starting training..."
python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=$CONFIG \
  data.num_workers=4 \
  grad_acc_steps=2 \
  steps=$STEPS \
  logging.freq=5 \
  logging.wandb.mode=online \
  logging.wandb.entity=XLM \
  logging.wandb.project=eeg \
  name=$NAME

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Check outputs:"
echo "  Directory: outputs/eeg_mae_base/"
echo "  Config: outputs/eeg_mae_base/config.yaml"
echo "  Metrics: outputs/eeg_mae_base/metrics.jsonl"
echo "  Logs: outputs/eeg_mae_base/train.log"
echo ""
echo "View loss:"
echo "  cat outputs/eeg_mae_base/metrics.jsonl | jq '.\"loss/out\"'"
echo ""
