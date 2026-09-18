# 重要修复总结

## 修复1: DataLoader数据分片 ✅

### 问题
之前的 `GroupedSampler` 没有真正分片数据：
- 每个GPU看到**相同的数据**
- 只是丢弃了不能被n_gpu整除的batch
- 没有使用rank信息

### 修复
```python
# 修改前
n_leftover_indices = len(indices) % self.n_gpu
self.indices = indices if n_leftover_indices == 0 else indices[:-n_leftover_indices]
return iter(self.indices)

# 修改后
# 1. 添加rank参数
def __init__(self, dataset, batch_size, drop_last, n_gpu, rank=0, mode="train"):
    self.rank = rank
    ...

# 2. 实际分片数据
n_leftover_indices = len(indices) % self.n_gpu
if n_leftover_indices != 0:
    indices = indices[:-n_leftover_indices]

# 每个rank获取自己的数据片段
self.indices = indices[self.rank::self.n_gpu]
```

### 验证
```bash
python verify_data_sharding.py
```

应该输出：
```
Rank 0: Total batches: 5
Rank 1: Total batches: 5  
Rank 2: Total batches: 5
Rank 3: Total batches: 5

✅ SUCCESS: Each rank has different data (properly sharded)
```

### 影响
- **修复前**: 4个GPU训练，实际有效batch = 150 × 2 × 1 = 300（每个GPU看相同数据）
- **修复后**: 4个GPU训练，实际有效batch = 150 × 2 × 4 = 1200 ✓

## 修复2: 启用token_avg与eeg_infra对齐 ✅

### 问题
- eeg_infra配置: `token_avg: True`
- ultra配置: `token_avg: false`
- Loss计算不一致

### 修复
```yaml
# configs/eeg/base.yaml 和 tiny.yaml
model:
  token_avg: true        # 启用
  token_avg_lambda: 0.1  # 与eeg_infra一致
```

### 实现确认
模型已有完整实现：
```python
# ultra/model_eeg.py
if self.token_avg and self.training and context is not None:
    # Attention pooling over all encoder outputs
    cls_output = self.cls_query_token.expand(b_size, -1, -1)
    attention_scores = torch.matmul(cls_output, key_value_tokens.transpose(-2, -1))
    attention_weights = F.softmax(attention_scores / math.sqrt(self.encoder.embed_dim), dim=-1)
    cls_token = torch.matmul(attention_weights, key_value_tokens).squeeze(1)
    pred_pixel_values_cls = self.cls_to_pixels(cls_token)
    loss_cls = F.mse_loss(pred_pixel_values_cls, masked_patches.mean(dim=1))
    
    # Combined loss
    loss = loss + self.token_avg_lambda * loss_cls
```

### 影响
- 辅助损失项，帮助encoder学习更好的全局表示
- 与eeg_infra训练完全对齐

## 当前训练配置检查清单

### ✅ 数据加载
- [x] Memmap加载
- [x] GroupedSampler按通道分组
- [x] **数据正确分片到各个rank**
- [x] On-the-fly normalization fallback
- [x] 16 workers + prefetch_factor=3

### ✅ 模型配置
- [x] MAE架构 (encoder: 512d/22层, decoder: 512d/8层)
- [x] 4D Fourier位置编码
- [x] **token_avg=true** (与eeg_infra一致)
- [x] Megatron初始化
- [x] FlashAttention (自动fallback)

### ✅ 训练配置
- [x] **有效batch size = 1200** (150 × 2 × 4)
- [x] 学习率: 2.4e-4 (与eeg_infra接近)
- [x] Gradient clipping: 5.0
- [x] Cosine LR调度
- [x] Warmup: 500步

### ✅ 分布式配置
- [x] FSDP (full_shard) - 当前脚本设置
- [x] 4 GPUs
- [x] **每个GPU看到不同数据**

## 与eeg_infra对照表

| 配置项 | eeg_infra | ultra (修复后) | 状态 |
|--------|-----------|----------------|------|
| 有效batch size | 1200 | 1200 | ✅ |
| token_avg | True | True | ✅ |
| token_avg_lambda | 0.1 | 0.1 | ✅ |
| 学习率 | 0.001 | 2.4e-4 | ⚠️ 不同 |
| Grad clip | 5.0 | 5.0 | ✅ |
| 数据分片 | 每GPU不同 | 每GPU不同 | ✅ |
| Masking ratio | 0.75 | 0.75 | ✅ |

### 学习率差异说明
- eeg_infra: lr=0.001 (AdamW)
- ultra: lr=2.4e-4 (AdamW)

建议：如果要完全对齐，修改为：
```bash
... optim.lr=0.001
```

## 测试建议

### 1. 验证数据分片
```bash
python verify_data_sharding.py
```

### 2. 对比训练曲线
运行相同步数，对比：
- Loss下降趋势
- 收敛速度
- 最终loss值

### 3. 检查WandB日志
确认记录的指标：
- `loss/out`: 总loss（包括token_avg部分）
- 应该与eeg_infra的loss可比

## 完整的训练命令（修复后）

```bash
bash test_train_4gpu.sh
```

配置：
- Config: `tiny.yaml`
- Steps: 10000
- Batch size: 150 per GPU
- Grad accumulation: 2
- GPUs: 4 (FSDP)
- **有效batch = 150 × 2 × 4 = 1200** ✓
- **数据已分片** ✓
- **token_avg已启用** ✓

## 预期结果

### 数据加载
```
Rank 0: Built EEG dataloader: 4926 samples, 5 batches
Rank 1: Built EEG dataloader: 4926 samples, 5 batches
Rank 2: Built EEG dataloader: 4926 samples, 5 batches
Rank 3: Built EEG dataloader: 4926 samples, 5 batches
```
- 每个rank看到总共4926个样本
- 但每个rank的5个batch是**不同的数据**
- 总共 5 × 4 = 20 batches/epoch

### Loss
```
step: 5   loss: ~0.75  # 包含token_avg辅助loss
step: 10  loss: ~0.73
step: 15  loss: ~0.72
...
```

应该与eeg_infra的loss趋势相似（因为：
- 相同的有效batch size
- 相同的token_avg配置
- 相同的模型架构
）

## 后续优化

如果要完全对齐eeg_infra：

1. **学习率**：改为0.001
   ```bash
   ... optim.lr=0.001
   ```

2. **Scheduler**: 检查是否需要调整
   - eeg_infra用trapezoid scheduler
   - ultra用cosine scheduler

3. **数据增强**: 检查是否有其他差异
