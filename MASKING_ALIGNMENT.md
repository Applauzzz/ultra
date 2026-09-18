# Block Masking 对齐说明

## eeg_infra vs ultra 对照

### eeg_infra 配置 (preprocessing/default.yaml)
```yaml
masking:
  use_block: True          # 使用block masking
  masking_window: 200
  masking_overlap: 20
  ratio: 0.55              # 注意：是0.55，不是0.75！
  radius_spat_mask: 0.03   # 空间masking半径
  radius_temp_mask: 3      # 时间masking半径
  dropout_ratio: 0.1       # Dropout比例
  dropout_radius: 0.04     # Dropout半径
```

### ultra 配置 (已修复)
```yaml
data:
  masking_ratio: 0.55        # ✅ 匹配
  masking_window: 200        # ✅ 匹配
  masking_overlap: 20        # ✅ 匹配
  use_block_masking: true    # ✅ 启用
  radius_spat_mask: 0.03     # ✅ 匹配
  radius_temp_mask: 3        # ✅ 匹配
  dropout_ratio: 0.1         # ✅ 匹配
  dropout_radius: 0.04       # ✅ 匹配

model:
  masking_ratio: 0.55        # ✅ 匹配
```

## Block Masking 实现

### 空间-时间块masking
```python
def create_block_masks(n_chans, masking_ratio, radius_spat_mask, radius_temp_mask,
                       num_patches, pos, dropout_ratio, dropout_ratio_radius):
    """
    创建block masks:
    1. 空间维度：使用spatial_masking选择相邻的通道
    2. 时间维度：将连续的时间patch组成块
    3. Dropout：额外的通道dropout
    """
    num_block_patches = num_patches // radius_temp_mask
    num_masked_chans = int(masking_ratio * n_chans)
    
    # 为每个时间块创建空间mask
    block_masks = [
        spatial_masking(pos, masking_ratio, radius_spat_mask)[:num_masked_chans]
        for _ in range(num_block_patches)
    ]
    
    # 将空间mask在时间维度重复radius_temp_mask次
    idx_block = np.repeat(
        np.array(block_masks)[:, np.newaxis, :], 
        radius_temp_mask, 
        axis=1
    ).reshape(-1, num_masked_chans)
    
    return masked_indices, unmasked_indices
```

### 与Random Masking的区别

**Random Masking (之前的默认)**:
```python
# 完全随机选择patches
num_patches = c * h
num_masks = int(masking_ratio * num_patches)
rand_indices = torch.rand(1, num_patches).argsort(dim=-1)
batch_mask = rand_indices[:, :num_masks]
```

**Block Masking (现在的配置)**:
```python
# 空间和时间上都有结构化的masking
# 1. 空间：mask掉空间上相邻的电极
# 2. 时间：在时间维度上连续mask
# 3. 更接近真实的信号丢失场景
```

## 影响分析

### 1. Masking Ratio变化
- **之前**: 0.75 (75%的patches被mask)
- **现在**: 0.55 (55%的patches被mask)
- **影响**: 
  - 模型看到更多的上下文信息
  - 重建任务略微简单
  - 可能收敛更快，但泛化能力需要评估

### 2. Block vs Random
- **Random**: 独立随机mask每个patch
- **Block**: 在空间和时间上连续mask
- **影响**:
  - Block更符合真实场景（传感器故障、信号中断）
  - 强制模型学习更强的空间-时间依赖
  - 可能提高鲁棒性

### 3. Dropout
- **Dropout ratio**: 0.1 (10%的通道)
- **Dropout radius**: 0.04
- **作用**: 额外的正则化，防止过拟合特定通道

## 验证Block Masking

### 测试脚本
```python
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

args = EEGDataArgs(
    data_path="/mnt/zehao/eeg_infra/data",
    batch_size=150,
    use_block_masking=True,
    masking_ratio=0.55,
    radius_spat_mask=0.03,
    radius_temp_mask=3,
)

dataloader, _ = build_eeg_dataloader(args, train=True)
eeg, pos, mask, unmask = next(iter(dataloader))

print(f"EEG shape: {eeg.shape}")
print(f"Masked indices shape: {mask.shape}")
print(f"Unmasked indices shape: {unmask.shape}")
print(f"Masking ratio: {mask.shape[0] / (mask.shape[0] + unmask.shape[0]):.2%}")
```

### 预期输出
```
EEG shape: torch.Size([150, C, T])
Masked indices shape: torch.Size([N_masked])
Unmasked indices shape: torch.Size([N_unmasked])
Masking ratio: 55.00%  # 应该接近0.55
```

## 完整对照表

| 配置项 | eeg_infra | ultra (修复前) | ultra (修复后) | 状态 |
|--------|-----------|----------------|----------------|------|
| use_block | True | False | True | ✅ |
| masking_ratio | 0.55 | 0.75 | 0.55 | ✅ |
| masking_window | 200 | 200 | 200 | ✅ |
| masking_overlap | 20 | 20 | 20 | ✅ |
| radius_spat_mask | 0.03 | - | 0.03 | ✅ |
| radius_temp_mask | 3 | - | 3 | ✅ |
| dropout_ratio | 0.1 | - | 0.1 | ✅ |
| dropout_radius | 0.04 | - | 0.04 | ✅ |

## 其他重要差异

### Window Duration
- **eeg_infra**: 2000 (2秒)
- **ultra base.yaml**: 用户已修改为2000 ✅
- **ultra tiny.yaml**: 10000 (10秒) ⚠️

### Clip Value
- **eeg_infra**: 15.0
- **ultra**: 10.0 ⚠️

建议统一：
```yaml
data:
  window_duration: 2000  # 2秒
  clip: 15.0             # 匹配eeg_infra
```

## 训练影响预测

### Loss变化
由于masking ratio从0.75降到0.55：
- 初始loss可能略低（更多上下文）
- 收敛可能更快
- 最终loss可能略好

### 训练动态
由于启用block masking：
- 梯度可能更稳定（结构化masking）
- 需要更多epoch来学习空间-时间依赖
- 模型鲁棒性可能提高

## 建议

### 立即启用
当前配置已经修复，可以直接训练：
```bash
bash test_train_4gpu.sh
```

### 进一步对齐
如果要100%匹配eeg_infra：

1. **Clip value**:
   ```yaml
   data.clip: 15.0
   ```

2. **Window duration**（tiny.yaml）:
   ```yaml
   data.window_duration: 2000
   ```

3. **验证mask分布**:
   运行验证脚本确认masking ratio实际为55%

### 监控指标
训练时关注：
- Loss是否与eeg_infra相似
- 收敛速度
- 模型对masked区域的重建质量
