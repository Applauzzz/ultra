# 完整配置对齐总结

## 已修复的关键问题

### ✅ 1. 数据分片 (Data Sharding)
- **问题**: 每个GPU看到相同数据
- **修复**: 添加rank参数，`indices[rank::n_gpu]`
- **影响**: 有效batch size从300提升到1200

### ✅ 2. Token Averaging
- **问题**: token_avg=false，与eeg_infra不一致
- **修复**: 启用token_avg=true, lambda=0.1
- **影响**: Loss计算包含辅助损失项

### ✅ 3. Block Masking
- **问题**: use_block_masking=false，使用random masking
- **修复**: 启用block masking + 全部参数
- **影响**: 更结构化的masking，更接近真实场景

### ✅ 4. Masking Ratio
- **问题**: masking_ratio=0.75
- **修复**: masking_ratio=0.55 (匹配eeg_infra)
- **影响**: 模型看到更多上下文

## 完整配置对照

| 配置项 | eeg_infra | ultra (修复前) | ultra (修复后) | 状态 |
|--------|-----------|----------------|----------------|------|
| **数据加载** |
| 数据分片 | 每GPU不同 | 全部相同 ❌ | 每GPU不同 ✅ | ✅ |
| window_duration | 2000 | 10000 | 2000 | ✅ |
| clip | 15.0 | 10.0 | 10.0 | ⚠️ |
| **Masking** |
| use_block_masking | True | False | True | ✅ |
| masking_ratio | 0.55 | 0.75 | 0.55 | ✅ |
| masking_window | 200 | 200 | 200 | ✅ |
| masking_overlap | 20 | 20 | 20 | ✅ |
| radius_spat_mask | 0.03 | - | 0.03 | ✅ |
| radius_temp_mask | 3 | - | 3 | ✅ |
| dropout_ratio | 0.1 | - | 0.1 | ✅ |
| dropout_radius | 0.04 | - | 0.04 | ✅ |
| **训练配置** |
| batch_size (per GPU) | 300 | 8 | 150 | ✅ |
| grad_acc_steps | 1 | 1 | 2 | ✅ |
| 有效batch size | 1200 | 32 ❌ | 1200 ✅ | ✅ |
| learning_rate | 0.001 | 2.4e-4 | 2.4e-4 | ⚠️ |
| grad_clip_norm | 5.0 | 5.0 | 5.0 | ✅ |
| **模型配置** |
| token_avg | True | False | True | ✅ |
| token_avg_lambda | 0.1 | 0.1 | 0.1 | ✅ |
| encoder_dim | 512 | 512 | 512 | ✅ |
| encoder_depth | 22 | 22 | 22 | ✅ |
| decoder_dim | 512 | 512 | 512 | ✅ |
| decoder_depth | 8 | 8 | 8 | ✅ |

## 剩余差异

### ⚠️ 需要考虑的差异

1. **Clip Value**
   - eeg_infra: 15.0
   - ultra: 10.0
   - **建议**: 改为15.0以完全对齐

2. **Learning Rate**
   - eeg_infra: 0.001
   - ultra: 2.4e-4
   - **建议**: 可以保持2.4e-4，或改为0.001

3. **Scheduler**
   - eeg_infra: trapezoid
   - ultra: cosine
   - **影响**: LR衰减曲线不同

## 当前配置性能预估

### 有效训练配置
```yaml
# 每GPU
batch_size: 150
grad_acc_steps: 2
num_gpus: 4

# 有效配置
有效batch = 150 × 2 × 4 = 1200 ✓
每个GPU看到不同的数据 ✓
```

### Masking配置
```yaml
# Block masking启用
use_block_masking: true
masking_ratio: 0.55  # 55%被mask

# 空间-时间结构化masking
radius_spat_mask: 0.03
radius_temp_mask: 3

# 额外正则化
dropout_ratio: 0.1
```

### 模型配置
```yaml
# Encoder
encoder_dim: 512
encoder_depth: 22
encoder_heads: 8

# Decoder
decoder_dim: 512
decoder_depth: 8

# Loss
token_avg: true  # 包含辅助loss
```

## 预期训练行为

### 与eeg_infra相比
- ✅ **相同**: 数据加载、masking策略、模型架构
- ✅ **相同**: 有效batch size、token averaging
- ⚠️ **不同**: learning rate、clip value

### Loss曲线
应该看到：
1. 初始loss ~0.6-0.8（由于masking_ratio=0.55）
2. 稳定下降
3. 收敛到 ~0.3-0.4
4. 与eeg_infra趋势相似

### 训练速度
```
步数/epoch: ~20 batches (8个recordings)
Step time: ~200-250ms (4 GPU, FSDP)
Throughput: ~2400 samples/sec
```

## 快速对齐checklist

运行训练前确认：

- [x] 数据分片启用 (rank参数)
- [x] Block masking启用
- [x] masking_ratio=0.55
- [x] token_avg=true
- [x] 有效batch size=1200
- [ ] 考虑是否修改clip=15.0
- [ ] 考虑是否修改lr=0.001

## 完全对齐配置（可选）

如果要100%匹配eeg_infra：

```yaml
# configs/eeg/base.yaml
data:
  clip: 15.0  # 改为15.0

optim:
  lr: 0.001   # 改为0.001
```

或命令行覆盖：
```bash
bash test_train_4gpu.sh
# 或
python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.clip=15.0 \
  optim.lr=0.001 \
  ...
```

## 验证方法

### 1. 数据分片
```bash
python verify_data_sharding.py
# 应该显示: ✅ Each rank has different data
```

### 2. Masking比例
```python
# 在训练中检查
print(f"Masked: {mask.shape}, Unmasked: {unmask.shape}")
print(f"Ratio: {mask.shape[0] / (mask.shape[0] + unmask.shape[0])}")
# 应该约等于 0.55
```

### 3. Loss对比
- 对比ultra和eeg_infra的loss曲线
- 趋势应该相似
- 数值可能有小差异（由于lr、scheduler不同）

## 总结

**核心修复（已完成）**：
1. ✅ 数据分片
2. ✅ Token averaging
3. ✅ Block masking
4. ✅ Masking ratio对齐

**可选调整**：
1. clip value: 10.0 → 15.0
2. learning rate: 2.4e-4 → 0.001

**当前状态**: 95%对齐，可以开始训练并对比结果！
