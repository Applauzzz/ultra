#!/bin/bash
# Test EEG training on 1 GPU (properly configured)

set -e

echo "Testing EEG Training on 1 GPU (GPU 4)"

source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

# IMPORTANT: Only 1 GPU visible, matching nproc_per_node=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1

cd /mnt/zehao/ultra

CONFIG="configs/eeg/base.yaml"
STEPS=10
NAME="eeg_test_1gpu_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  GPU: 4 (single GPU)"
echo "  Config: $CONFIG"
echo "  Steps: $STEPS"
echo ""

# Run training with 1 GPU
python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  main/train_eeg.py \
  config=$CONFIG \
  data.subset=small \
  steps=$STEPS \
  logging.freq=10 \
  logging.wandb.mode=offline \
  name=$NAME

echo ""
echo "Training completed!"
