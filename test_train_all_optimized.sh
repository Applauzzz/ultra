#!/bin/bash
set -e

echo "Testing ultra with 'all' dataset (optimized lazy loading)"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export DATA_PATH=/mnt_upfs/zehao/database/egg/reve_official_data

cd /mnt/zehao/ultra

# Start with moderate workers
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  main/train_eeg.py \
  --config configs/eeg/base.yaml \
  --data.data_path "$DATA_PATH" \
  --data.subset all \
  --data.num_workers 4 \
  --data.prefetch_factor 2 \
  --data.persistent_workers false \
  --steps 50 \
  --log_freq 10 \
  --dump_dir outputs/eeg_mae_all_test

echo "Training completed"
