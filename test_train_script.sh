#!/bin/bash
# Quick test script for EEG training
# Runs a few steps to verify everything works

set -e

echo "========================================"
echo "Testing EEG Training Script"
echo "========================================"

# Configuration
CONFIG="configs/eeg/tiny.yaml"
STEPS=5
NAME="test_train_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Steps: $STEPS"
echo "  Name: $NAME"
echo ""

# Run training
echo "Starting training..."
python main/train_eeg.py \
  config=$CONFIG \
  distributed.dp_shard=1 \
  steps=$STEPS \
  logging.freq=1 \
  logging.wandb.log=false \
  name=$NAME \
  2>&1 | tee test_train.log

echo ""
echo "========================================"
echo "Training test completed!"
echo "========================================"
echo ""
echo "Check outputs:"
echo "  Config: outputs/$NAME/config.yaml"
echo "  Metrics: outputs/$NAME/metrics.jsonl"
echo "  Logs: outputs/$NAME/train.log"
echo ""
