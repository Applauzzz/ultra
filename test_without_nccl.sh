#!/bin/bash
# Test if removing NCCL environment variables allows workers to function
set -e

echo "========================================"
echo "Testing WITHOUT NCCL Environment Variables"
echo "========================================"

cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

# Only set CUDA devices - NO NCCL variables!
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo ""
echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  NCCL_IB_HCA: ${NCCL_IB_HCA:-<not set>}"
echo "  NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-<not set>}"
echo "  GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-<not set>}"
echo ""

CONFIG="configs/eeg/base.yaml"
STEPS=10
NAME="test_no_nccl_$(date +%Y%m%d_%H%M%S)"

echo "Starting training with:"
echo "  Config: $CONFIG"
echo "  Workers: 2"
echo "  Steps: $STEPS"
echo ""

# Run with 2 workers (should work if NCCL env vars were the problem)
timeout 180 python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=$CONFIG \
  data.num_workers=2 \
  grad_acc_steps=2 \
  steps=$STEPS \
  logging.freq=5 \
  logging.wandb.mode=offline \
  name=$NAME

echo ""
echo "========================================"
echo "✅ SUCCESS - Workers function without NCCL env vars!"
echo "========================================"
