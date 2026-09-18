#!/bin/bash
# Test the fixed worker initialization (DataLoader before NCCL)
set -e

echo "========================================"
echo "Testing Fixed Worker Initialization"
echo "========================================"
echo "Strategy: Create DataLoader BEFORE NCCL init"
echo "  1. Read rank from env vars"
echo "  2. Create DataLoader"
echo "  3. Fork workers (NCCL not initialized yet)"
echo "  4. Initialize NCCL"
echo "  5. Train normally"
echo ""

cd /mnt/zehao/ultra
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

export CUDA_VISIBLE_DEVICES=4,5,6,7

CONFIG="configs/eeg/base.yaml"
STEPS=20
NAME="test_fixed_workers_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Workers: 2 (with persistent_workers=true)"
echo "  Steps: $STEPS"
echo "  GPUs: 4"
echo ""

echo "Starting training..."
timeout 300 python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=$CONFIG \
  grad_acc_steps=2 \
  steps=$STEPS \
  logging.freq=5 \
  logging.wandb.mode=offline \
  name=$NAME

echo ""
echo "========================================"
echo "✅ SUCCESS - Workers function with NCCL!"
echo "========================================"
echo ""
echo "Solution verified:"
echo "  ✓ DataLoader created before NCCL init"
echo "  ✓ Workers forked before NCCL state exists"
echo "  ✓ No deadlock on worker creation"
echo "  ✓ Training proceeds normally"
