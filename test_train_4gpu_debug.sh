#!/bin/bash
set -e

echo "Starting ultra training test (4 GPU, reduced workers)"
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DATA_PATH=/mnt_upfs/zehao/database/egg/reve_official_data

cd /mnt/zehao/ultra

# Run with reduced workers to avoid deadlock
torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  main/train_eeg.py \
  --config configs/eeg/base.yaml \
  --data.data_path "$DATA_PATH" \
  --data.num_workers 2 \
  --data.prefetch_factor 2 \
  --data.persistent_workers false \
  --steps 100 \
  --dump_dir outputs/eeg_mae_debug

echo "Training completed"
