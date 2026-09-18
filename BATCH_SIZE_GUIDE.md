# Batch Size 配置对照指南

## eeg_infra vs ultra 框架对比

### eeg_infra (原框架)
```yaml
trainer:
  batch_size: 300
  accumulate_grad_batches: 1
  n_gpus: 4
```
**有效batch size** = `300 × 1 × 4 = 1200`

### ultra框架
需要设置两个参数来达到相同的有效batch size。

## 有效Batch Size计算公式

```
有效Batch Size = per_GPU_batch_size × grad_acc_steps × num_GPUs
```

## 推荐配置方案

### 方案1：直接匹配（可能OOM）
```yaml
data.batch_size: 300
grad_acc_steps: 1
distributed.dp_replicate: 4
```
**有效batch = 300 × 1 × 4 = 1200** ✓

**优点**：配置简单，直接对应
**缺点**：可能显存不足（特别是Base/Large模型）

### 方案2：梯度累积（推荐）
```yaml
data.batch_size: 150
grad_acc_steps: 2
distributed.dp_replicate: 4
```
**有效batch = 150 × 2 × 4 = 1200** ✓

**优点**：
- 相同的有效batch size
- 降低显存占用（每步只需150样本）
- 更稳定，不易OOM

**缺点**：
- 训练稍慢（需要2次backward才更新一次参数）

### 方案3：更激进的累积（OOM时使用）
```yaml
data.batch_size: 100
grad_acc_steps: 3
distributed.dp_replicate: 4
```
**有效batch = 100 × 3 × 4 = 1200** ✓

### 方案4：小batch调试
```yaml
data.batch_size: 64
grad_acc_steps: 1
distributed.dp_replicate: 4
```
**有效batch = 64 × 1 × 4 = 256** ❌ (仅用于调试)

## 不同模型的建议配置

### Tiny模型 (~13M参数)
```yaml
# 选项1: 无梯度累积
data.batch_size: 512
grad_acc_steps: 1
# 有效batch = 512 × 1 × 4 = 2048

# 选项2: 匹配eeg_infra
data.batch_size: 300
grad_acc_steps: 1
# 有效batch = 300 × 1 × 4 = 1200
```

### Base模型 (~94M参数) - DDP模式
```yaml
# 推荐：梯度累积
data.batch_size: 150
grad_acc_steps: 2
# 有效batch = 150 × 2 × 4 = 1200

# 如果还OOM：
data.batch_size: 100
grad_acc_steps: 3
# 有效batch = 100 × 3 × 4 = 1200
```

### Large模型 (~200M参数)
```yaml
# 需要FSDP或更激进的累积
data.batch_size: 64
grad_acc_steps: 5
# 有效batch = 64 × 5 × 4 = 1280 (接近1200)
```

## 验证配置是否正确

### 1. 检查日志中的有效batch size
训练日志会显示：
```
optim/total_samples: 1200  # 每次optimizer step处理的总样本数
```

### 2. 对比学习率调度
如果有效batch size匹配，学习率曲线应该相似。

### 3. 对比训练速度
- **eeg_infra**: ~X samples/sec
- **ultra**: 应该接近相同的throughput

## 常见错误

### ❌ 错误1: 忽略梯度累积
```yaml
data.batch_size: 300  # 只看这个
# 实际有效batch = 300 × 1 × 4 = 1200
```

### ❌ 错误2: batch size太小
```yaml
data.batch_size: 8
grad_acc_steps: 1
# 有效batch = 8 × 1 × 4 = 32 (远小于1200！)
```

### ❌ 错误3: 没有考虑GPU数量
```yaml
# 单GPU
data.batch_size: 300
grad_acc_steps: 4
# 有效batch = 300 × 4 × 1 = 1200 ✓

# 4 GPU (需要调整!)
data.batch_size: 300
grad_acc_steps: 4
# 有效batch = 300 × 4 × 4 = 4800 ❌ (太大了！)
```

## 快速对照表

| eeg_infra | ultra (4 GPU DDP) | 有效Batch | GPU内存 |
|-----------|-------------------|-----------|---------|
| bs=300, acc=1 | bs=300, acc=1 | 1200 | 高 |
| bs=300, acc=1 | bs=150, acc=2 | 1200 | 中 ⭐ |
| bs=300, acc=1 | bs=100, acc=3 | 1200 | 低 |
| bs=300, acc=1 | bs=75, acc=4  | 1200 | 很低 |

⭐ = 推荐配置

## 性能影响

### 梯度累积的代价
- **计算时间**: 几乎无影响（只是多次backward）
- **通信开销**: 略微增加（但在DDP下影响很小）
- **内存节省**: 显著（可以避免OOM）

### 何时使用梯度累积
- ✅ GPU内存不足时
- ✅ 想要更大的有效batch size但单GPU放不下
- ✅ 需要在不同硬件上保持相同的训练配置

### 何时不用梯度累积
- ✅ 内存充足
- ✅ 追求最快训练速度
- ✅ 使用FSDP已经解决内存问题

## 迁移检查清单

从eeg_infra迁移到ultra时：

- [ ] 计算原框架的有效batch size
- [ ] 根据GPU内存选择合适的per-GPU batch size
- [ ] 设置grad_acc_steps以匹配有效batch size
- [ ] 验证训练日志中的total_samples
- [ ] 对比loss曲线是否相似
- [ ] 检查训练速度（samples/sec）

## 示例：完整迁移

### eeg_infra配置
```yaml
trainer:
  batch_size: 300
  accumulate_grad_batches: 1
  n_gpus: 4
  lr: 0.001
```

### ultra配置（推荐）
```bash
python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.batch_size=150 \
  grad_acc_steps=2 \
  optim.lr=0.001 \
  distributed.dp_replicate=4
```

**验证**:
- 有效batch: 150 × 2 × 4 = 1200 ✓
- 学习率: 0.001 ✓
- GPU数: 4 ✓
