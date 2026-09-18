#!/bin/bash
# Test with small subset first to isolate the issue

ssh 10.60.137.131 'bash -s' << 'REMOTE'
set -e

export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /mnt/zehao/ultra

source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

echo "Testing with SMALL subset (should work)..."

timeout 180 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.data_path=/mnt_upfs/zehao/database/egg/reve_official_data \
  data.subset=small \
  data.num_workers=0 \
  steps=20 \
  logging.freq=2 \
  name=test_small 2>&1 | tee test_small.log

if [ $? -eq 0 ]; then
    echo "✓ Small subset works!"
    echo "Now testing ALL subset with num_workers=0..."
    
    timeout 600 python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
      main/train_eeg.py \
      config=configs/eeg/base.yaml \
      data.data_path=/mnt_upfs/zehao/database/egg/reve_official_data \
      data.subset=all \
      data.num_workers=0 \
      steps=10 \
      logging.freq=2 \
      name=test_all_noworkers 2>&1 | tee test_all_noworkers.log
    
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        echo "✓✓ ALL subset works with num_workers=0!"
        echo "Issue is with DataLoader workers."
    elif [ $EXIT -eq 124 ]; then
        echo "✗ Timed out - probably hanging"
        tail -30 test_all_noworkers.log
    else
        echo "✗ Failed with exit code $EXIT"
        tail -30 test_all_noworkers.log
    fi
else
    echo "✗ Even small subset failed"
    tail -30 test_small.log
fi
REMOTE
