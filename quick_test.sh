#!/bin/bash
# Quick validation test - 10 steps on 1 GPU

# Activate conda environment
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

CUDA_VISIBLE_DEVICES=4 python main/train_eeg.py \
  config=configs/eeg/tiny.yaml \
  distributed.dp_shard=1 \
  steps=10 \
  logging.freq=1 \
  name=quick_validation
