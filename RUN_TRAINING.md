# EEG MAE 训练快速启动指南

## 快速测试（4 GPU）

### 使用GPU 4-7进行100步测试

```bash
# 直接运行脚本
./test_train_4gpu.sh
```

## 手动启动选项

### 1. 单GPU测试（快速验证）

```bash
CUDA_VISIBLE_DEVICES=4 python main/train_eeg.py \
  config=configs/eeg/tiny.yaml \
  distributed.dp_shard=1 \
  steps=100 \
  name=test_single_gpu
```

### 2. 2 GPU训练

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
  main/train_eeg.py \
  config=configs/eeg/tiny.yaml \
  distributed.dp_shard=2 \
  steps=1000 \
  name=test_2gpu
```

### 3. 4 GPU标准训练（推荐）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  distributed.dp_shard=4 \
  steps=10000 \
  name=eeg_base_production
```

### 4. 4 GPU完整训练（带WandB）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  distributed.dp_shard=4 \
  steps=100000 \
  logging.wandb.mode=online \
  logging.wandb.entity=YOUR_WANDB_ENTITY \
  checkpoint.dump.every=5000 \
  name=eeg_mae_full
```

## 配置文件选择

| 配置 | 模型 | 参数量 | 推荐GPU | Batch | 说明 |
|------|------|--------|---------|-------|------|
| `tiny.yaml` | 256d, 12层 | ~13M | 1-2 | 512 | 快速实验 |
| `base.yaml` | 512d, 22层 | ~94M | 4 | 300 | 标准训练 |
| `large.yaml` | 768d, 24层 | ~200M | 8 | 128 | 最大性能 |

## 常用参数覆盖

```bash
# 修改学习率
... optim.lr=1e-4

# 修改batch size
... data.batch_size=256

# 修改训练步数
... steps=50000

# 修改checkpoint频率
... checkpoint.dump.every=2000

# 启用WandB
... logging.wandb.mode=online logging.wandb.entity=your_entity

# 修改数据路径
... data.data_path=/your/data/path
```

## 监控训练

### 实时查看日志

```bash
tail -f outputs/eeg_mae_base/train.log
```

### 查看loss曲线

```bash
cat outputs/eeg_mae_base/metrics.jsonl | jq '.["loss/out"]'
```

### 查看学习率

```bash
cat outputs/eeg_mae_base/metrics.jsonl | jq '.["optim/lr"]'
```

### 查看完整指标

```bash
cat outputs/eeg_mae_base/metrics.jsonl | jq '.' | less
```

### 监控GPU

```bash
watch -n 1 nvidia-smi
```

## 预期性能（Base模型，4 GPU）

```
Step time: ~150-200ms/step
Throughput: ~1500-2000 samples/sec
Memory: ~15-20GB/GPU
Loss: 开始 ~1.0, 收敛到 ~0.3-0.5

训练时间估算:
  - 100 steps: ~20-30秒
  - 1k steps: ~3-5分钟
  - 10k steps: ~30-40分钟
  - 100k steps: ~5-7小时
```

## Checkpoint管理

### 从checkpoint恢复

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  name=eeg_mae_base  # 使用相同的name会自动恢复
```

### 从特定checkpoint初始化

```bash
... checkpoint.init_ckpt_path=/path/to/checkpoint/step_10000
```

## 常见问题

### GPU OOM

```bash
# 减小batch size
... data.batch_size=128

# 或启用梯度累积
... grad_acc_steps=2 data.batch_size=150
```

### 训练速度慢

```bash
# 增加workers
... data.num_workers=24

# 增加prefetch
... data.prefetch_factor=4
```

### Loss不收敛

```bash
# 调整学习率
... optim.lr=2e-4

# 增加warmup
... optim.warmup=1000
```

## 后台运行

使用nohup在后台运行：

```bash
nohup ./test_train_4gpu.sh > train.out 2>&1 &

# 查看输出
tail -f train.out
```

使用screen：

```bash
screen -S eeg_train
./test_train_4gpu.sh
# Ctrl+A, D detach

# 重新连接
screen -r eeg_train
```

## 完整示例

```bash
# 在后台运行完整训练
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  distributed.dp_shard=4 \
  steps=100000 \
  optim.lr=2.4e-4 \
  checkpoint.dump.every=5000 \
  logging.freq=100 \
  logging.wandb.mode=online \
  logging.wandb.entity=your_entity \
  name=eeg_mae_production_run1 \
  > train.log 2>&1 &

# 监控训练
tail -f train.log
watch -n 1 nvidia-smi
```
