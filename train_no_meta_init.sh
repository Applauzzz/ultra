#!/bin/bash
# Train with meta_init disabled to avoid hang

ssh 10.60.137.131 'bash -s' << 'REMOTE'
set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/zehao/ultra

source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

echo "Starting training with meta_init=false..."

python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.data_path=/mnt_upfs/zehao/database/egg/reve_official_data \
  data.subset=all \
  data.num_workers=2 \
  data.prefetch_factor=2 \
  data.persistent_workers=false \
  model.meta_init=false \
  steps=100 \
  logging.freq=5 \
  logging.wandb.mode=online \
  logging.wandb.entity=XLM \
  logging.wandb.project=eeg \
  name=eeg_all_no_meta_$(date +%Y%m%d_%H%M%S) 2>&1 | tee train_no_meta.log

echo "Training completed!"
REMOTE
