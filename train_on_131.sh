#!/bin/bash
# Direct training script for 10.60.137.131
# Usage: ssh到10.60.137.131后直接运行此脚本

set -e

echo "=========================================="
echo "Ultra Training on 10.60.137.131"
echo "=========================================="

# 环境设置
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DATA_PATH=/mnt_upfs/zehao/database/egg/reve_official_data

# 激活环境
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2  # 或根据实际环境修改

cd /mnt/zehao/ultra

echo ""
echo "Environment Check:"
echo "  Host: $(hostname)"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  GPUs visible: $CUDA_VISIBLE_DEVICES"
echo ""

# 检查GPU状态
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | grep -E "^[4567],"
echo ""

# 确认开始
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo "Starting training..."
echo ""

# 运行训练 (使用懒加载优化后的代码)
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  main/train_eeg.py \
  --config configs/eeg/base.yaml \
  --data.data_path "$DATA_PATH" \
  --data.subset all \
  --data.num_workers 2 \
  --data.prefetch_factor 2 \
  --data.persistent_workers false \
  --steps 1000 \
  --log_freq 50 \
  --checkpoint.interval 100 \
  --dump_dir outputs/eeg_mae_all_$(date +%Y%m%d_%H%M%S)

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
