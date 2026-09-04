# EEG MAE Training Guide

完整的EEG MAE训练指南，包括单GPU、多GPU、配置说明等。

## 快速开始

### 1. 环境准备

```bash
cd /mnt/zehao/ultra
conda activate ultra_2  # 或其他有PyTorch + scipy的环境
```

### 2. 单GPU训练（快速测试）

```bash
# 使用Tiny模型快速测试
python main/train_eeg.py \
  config=configs/eeg/tiny.yaml \
  distributed.dp_shard=1 \
  steps=100 \
  name=test_tiny
```

### 3. 多GPU训练（推荐）

```bash
# 4 GPU训练 - Base模型
torchrun --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  name=eeg_base_4gpu

# 8 GPU训练 - Large模型
torchrun --nproc_per_node=8 \
  main/train_eeg.py \
  config=configs/eeg/large.yaml \
  name=eeg_large_8gpu
```

## 配置文件

### 可用配置

| 配置文件 | 模型大小 | 参数量 | 推荐GPU数 | Batch Size | 训练时间估计 |
|---------|---------|--------|-----------|-----------|------------|
| `tiny.yaml` | 256d, 12层 | ~20M | 1-2 | 512 | ~2小时/epoch |
| `base.yaml` | 512d, 22层 | ~94M | 4 | 300 | ~4小时/epoch |
| `large.yaml` | 768d, 24层 | ~200M | 8 | 128 | ~8小时/epoch |

### 配置文件位置

```
configs/eeg/
├── tiny.yaml     # 快速实验
├── base.yaml     # 标准配置（推荐）
└── large.yaml    # 最大性能
```

## 命令行覆盖

可以通过命令行覆盖配置文件中的任何参数：

```bash
# 修改学习率
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  optim.lr=1e-4

# 修改batch size
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.batch_size=128

# 修改训练步数
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  steps=5000

# 修改数据路径
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.data_path=/path/to/your/data

# 组合多个覆盖
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  optim.lr=1e-4 \
  data.batch_size=256 \
  steps=10000 \
  name=custom_experiment
```

## 分布式训练

### FSDP配置

```yaml
distributed:
  dp_replicate: 1
  dp_shard: 4      # FSDP分片数（通常等于GPU数）
  tp_size: 1       # Tensor并行（保持为1）
  sp_size: 1       # Sequence并行（保持为1）
  fsdp_type: full_shard  # full_shard, shard_grad_op, no_shard
```

### FSDP类型说明

- **`full_shard`** (推荐): 完全分片参数、梯度、优化器状态
  - 最省内存
  - 可训练最大的模型

- **`shard_grad_op`**: 分片梯度和优化器状态，参数不分片
  - 中等内存使用
  - 通信开销较小

- **`no_shard`**: 不分片（等同于DDP）
  - 内存使用最大
  - 通信最快

### 多节点训练

```bash
# Node 0 (master)
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=<master_ip> \
  --master_port=29500 \
  main/train_eeg.py \
  config=configs/eeg/large.yaml

# Node 1
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=<master_ip> \
  --master_port=29500 \
  main/train_eeg.py \
  config=configs/eeg/large.yaml
```

## Checkpoint管理

### 自动保存

```yaml
checkpoint:
  dump:
    every: 1000  # 每1000步保存一次
  keep_latest: 3  # 保留最新3个checkpoint
```

### 从Checkpoint恢复

```bash
# 自动从最新checkpoint恢复
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  checkpoint.path=outputs/eeg_mae_base/checkpoints

# 从特定checkpoint初始化
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  checkpoint.init_ckpt_path=/path/to/checkpoint
```

### Checkpoint文件结构

```
outputs/eeg_mae_base/
├── checkpoints/
│   ├── step_1000/
│   ├── step_2000/
│   └── step_3000/
├── config.yaml
├── metrics.jsonl
├── model_structure.txt
└── train.log
```

## WandB日志

### 启用WandB

```yaml
logging:
  wandb:
    project: eeg_mae
    entity: your_entity  # 你的WandB entity
    log: true
    tags: ["eeg", "mae", "base"]
```

### 通过命令行配置

```bash
# 启用WandB
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  logging.wandb.log=true \
  logging.wandb.entity=your_entity

# 离线模式
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  logging.wandb.log=true \
  logging.wandb.offline=true
```

### 记录的指标

- `loss/out`: 重建损失
- `optim/lr`: 学习率
- `optim/grad_norm`: 梯度范数
- `speed/samples_per_sec`: 吞吐量（samples/秒）
- `speed/curr_iter_time`: 每步时间
- `memory/max_active_pct`: GPU内存使用率

## 性能优化

### 数据加载优化

```yaml
data:
  num_workers: 16        # 增加worker数量
  prefetch_factor: 3     # 增加预取factor
  persistent_workers: true  # 启用persistent workers
```

### GPU内存优化

```bash
# 减小batch size
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.batch_size=128

# 启用梯度累积
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  grad_acc_steps=2 \
  data.batch_size=150  # 有效batch = 150 * 2 = 300
```

### 训练速度优化

```yaml
env:
  enable_tf32: true  # 启用TF32加速（A100）
```

## 常见问题

### Q: GPU内存不足 (OOM)

**解决方案**：

1. 减小batch size：
   ```bash
   data.batch_size=128
   ```

2. 启用梯度累积：
   ```bash
   grad_acc_steps=2
   ```

3. 减小模型大小：
   ```bash
   config=configs/eeg/tiny.yaml
   ```

4. 使用更多GPU（FSDP）：
   ```bash
   distributed.dp_shard=8
   ```

### Q: 训练速度慢

**解决方案**：

1. 增加数据加载workers：
   ```bash
   data.num_workers=16
   ```

2. 增加prefetch factor：
   ```bash
   data.prefetch_factor=4
   ```

3. 确保数据在SSD/NVMe上

4. 启用persistent workers：
   ```bash
   data.persistent_workers=true
   ```

### Q: Loss不收敛

**检查列表**：

1. 学习率是否合适：
   ```bash
   optim.lr=2.4e-4  # Base模型推荐值
   ```

2. 是否需要更长的warmup：
   ```bash
   optim.warmup_steps=1000
   ```

3. 检查梯度范数：
   ```bash
   # 在日志中查看 optim/grad_norm
   # 如果过大（>10），可能需要调整学习率
   ```

### Q: 如何继续训练

```bash
# 自动从最新checkpoint恢复
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  name=same_experiment_name  # 使用相同的name

# Checkpoint会自动从 outputs/same_experiment_name/checkpoints 加载
```

## 监控训练

### 查看日志

```bash
# 实时查看训练日志
tail -f outputs/eeg_mae_base/train.log

# 查看指标
cat outputs/eeg_mae_base/metrics.jsonl | jq .
```

### GPU监控

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi
```

### 训练进度

```python
# 解析metrics.jsonl
import json

with open('outputs/eeg_mae_base/metrics.jsonl') as f:
    for line in f:
        metrics = json.loads(line)
        print(f"Step {metrics['global_step']}: "
              f"loss={metrics['loss/out']:.4f}, "
              f"lr={metrics['optim/lr']:.2e}")
```

## 完整示例

### 示例1: 快速验证（单GPU，100步）

```bash
python main/train_eeg.py \
  config=configs/eeg/tiny.yaml \
  distributed.dp_shard=1 \
  steps=100 \
  logging.freq=10 \
  name=quick_test
```

### 示例2: 标准训练（4 GPU，10k步）

```bash
torchrun --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  steps=10000 \
  checkpoint.dump.every=1000 \
  logging.wandb.log=true \
  logging.wandb.entity=your_entity \
  name=eeg_base_production
```

### 示例3: 大规模训练（8 GPU，20k步，Large模型）

```bash
torchrun --nproc_per_node=8 \
  main/train_eeg.py \
  config=configs/eeg/large.yaml \
  steps=20000 \
  grad_acc_steps=2 \
  checkpoint.dump.every=2000 \
  logging.wandb.log=true \
  logging.wandb.entity=your_entity \
  name=eeg_large_full
```

## 预期性能

### Base模型（4 GPU）

```
Step time: ~150-200ms/step
Throughput: ~1500-2000 samples/sec
Memory: ~15-20GB/GPU
Loss: 开始 ~1.0, 收敛到 ~0.3-0.5
```

### Large模型（8 GPU）

```
Step time: ~300-400ms/step
Throughput: ~800-1000 samples/sec
Memory: ~40-50GB/GPU
Loss: 开始 ~1.0, 收敛到 ~0.2-0.4
```

## 下一步

训练完成后：

1. **评估模型**: 
   - 从checkpoint加载encoder
   - 在下游任务上fine-tune

2. **导出模型**:
   ```python
   from ultra.model_eeg import build_mae_model
   
   model = build_mae_model(config)
   model.load_state_dict(checkpoint['model'])
   
   # 保存encoder
   torch.save(model.encoder.state_dict(), 'encoder.pth')
   ```

3. **HuggingFace Hub**:
   - 转换为HF格式
   - 上传到hub

## 支持

遇到问题？

1. 检查日志：`outputs/<name>/train.log`
2. 查看本文档的常见问题部分
3. 检查集成测试：`python test_eeg_integration.py`
