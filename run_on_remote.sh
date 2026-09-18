#!/bin/bash
# Run ultra training on remote server 10.60.137.131 with free GPUs

set -e

REMOTE_HOST="10.60.137.131"
REMOTE_USER="ubuntu"  # 修改为实际用户名
REMOTE_GPU="4,5,6,7"
PROJECT_DIR="/mnt/zehao/ultra"  # 远程服务器上的项目路径

echo "=========================================="
echo "Running ultra training on remote server"
echo "=========================================="
echo "Remote: $REMOTE_USER@$REMOTE_HOST"
echo "GPUs: $REMOTE_GPU"
echo ""

# SSH到远程服务器并运行训练
ssh $REMOTE_USER@$REMOTE_HOST << 'REMOTE_SCRIPT'
set -e

# 环境设置
export CUDA_VISIBLE_DEVICES=4,5,6,7
export DATA_PATH=/mnt_upfs/zehao/database/egg/reve_official_data

cd /mnt/zehao/ultra

# 激活环境
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2  # 或 eeg-infra

echo "Environment:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# 检查GPU空闲情况
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo ""
echo "Starting training..."
echo ""

# 运行训练
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
  --dump_dir outputs/eeg_mae_all_remote

echo ""
echo "Training completed!"
REMOTE_SCRIPT

echo ""
echo "=========================================="
echo "Remote training finished"
echo "=========================================="
