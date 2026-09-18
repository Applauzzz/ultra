#!/bin/bash
# Test with checkpoint loading temporarily disabled

set -e

cd /mnt/zehao/ultra

source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

export CUDA_VISIBLE_DEVICES=4
export CUDA_LAUNCH_BLOCKING=1

echo "Testing with checkpoint loading bypassed..."

# Temporarily patch train_eeg.py to skip checkpoint.load
cp main/train_eeg.py main/train_eeg.py.backup

# Comment out checkpoint.load line
sed -i 's/checkpoint.load(model, optimizer, train_state, world_mesh)/# checkpoint.load(model, optimizer, train_state, world_mesh)  # TEMPORARILY DISABLED/' main/train_eeg.py

# Run training
timeout 180 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.subset=small \
  steps=5 \
  logging.freq=1 \
  logging.wandb.mode=offline \
  name=test_no_ckpt 2>&1 | tee test_no_checkpoint.log

EXIT_CODE=$?

# Restore original file
mv main/train_eeg.py.backup main/train_eeg.py

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training succeeded without checkpoint.load!"
    echo "Problem confirmed: checkpoint.load() causes hang"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "✗ Still timed out"
    tail -30 test_no_checkpoint.log
else
    echo "✗ Failed with exit code $EXIT_CODE"
    tail -30 test_no_checkpoint.log
fi
