#!/bin/bash
# Test 4 GPU + all subset with reduced workers to avoid hang

set -e

echo "========================================"
echo "Testing 4 GPU + All Subset (Fixed)"
echo "========================================"

source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

cd /mnt/zehao/ultra

echo ""
echo "Configuration:"
echo "  GPUs: 4,5,6,7 (4 GPUs)"
echo "  Subset: all"
echo "  Workers: 0 (main process only, to avoid hang)"
echo ""

python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.subset=all \
  data.num_workers=1 \
  data.batch_size=16 \
  steps=50 \
  logging.freq=5 \
  logging.wandb.mode=offline \
  name=eeg_4gpu_all_fixed_$(date +%Y%m%d_%H%M%S)

echo ""
echo "Training completed!"
echo ""
echo "If this works, the issue is with DataLoader workers in 4 GPU + all config."
echo "Solution: Use num_workers=0 or 1 for all subset with 4 GPUs."
