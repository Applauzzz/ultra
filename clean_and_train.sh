#!/bin/bash
# Clean old checkpoints and start fresh training

set -e

echo "========================================"
echo "Clean Training Run - 4 GPUs"
echo "========================================"

# Activate conda environment
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0

# Use GPUs 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Clean old outputs
echo "Cleaning old checkpoints and outputs..."
rm -rf outputs/eeg_mae_base/checkpoints/*
rm -f outputs/eeg_mae_base/metrics.jsonl

# Configuration
CONFIG="configs/eeg/base.yaml"
STEPS=1000
NAME="eeg_clean_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  Python: $(which python)"
echo "  GPUs: 4,5,6,7"
echo "  Config: $CONFIG"
echo "  Steps: $STEPS"
echo "  Name: $NAME"
echo ""

# Run training with 4 GPUs (DDP mode)
echo "Starting fresh training..."
python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=$CONFIG \
  distributed.dp_replicate=4 \
  distributed.dp_shard=1 \
  distributed.fsdp_type=no_shard \
  data.batch_size=512 \
  steps=$STEPS \
  logging.freq=10 \
  logging.wandb.mode=online \
  logging.wandb.entity=XLM \
  logging.wandb.project=eeg \
  name=$NAME

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Check results:"
echo "  Metrics: cat outputs/eeg_mae_base/metrics.jsonl | jq '.\"loss/out\"'"
echo "  Loss curve: cat outputs/eeg_mae_base/metrics.jsonl | jq '[.global_step, .\"loss/out\"]'"
echo ""
