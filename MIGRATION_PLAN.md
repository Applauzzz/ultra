# EEG训练Pipeline迁移到Ultra框架方案

## 框架对比分析

### 1. **训练框架差异**

| 特性 | eeg_infra | ultra |
|------|-----------|-------|
| **分布式框架** | Accelerate | PyTorch DDP + FSDP + TP + SP |
| **配置系统** | Hydra | Hydra (OmegaConf) |
| **数据格式** | NumPy memmap | JSONL (文本) |
| **模型类型** | MAE (Masked Autoencoder) | Transformer LLM |
| **训练目标** | 重建masked patches | Next token prediction |
| **数据加载** | 自定义Dataset + GroupedSampler | JSONL iterator + token packing |
| **并行策略** | DDP (Accelerate自动) | FSDP + TP + SP (手动配置) |

### 2. **核心差异点**

#### **数据加载差异**
```python
# eeg_infra: Memmap + 窗口采样
eeg_data = np.memmap('recording.npy', shape=(T, C))
window = eeg_data[offset:offset+10000]

# ultra: JSONL + tokenization
with open('data.jsonl') as f:
    text = json.loads(f.readline())['text']
    tokens = tokenizer.encode(text)
```

#### **模型差异**
```python
# eeg_infra: MAE forward
loss = mae(x, pos, batch_mask, batch_unmask)  # 重建loss

# ultra: LLM forward  
output = model(input_ids=input_ids, labels=labels)  # CE loss
loss = output.loss
```

#### **训练循环差异**
```python
# eeg_infra: Accelerate自动处理
with accelerator.accumulate(model):
    loss = model(x, pos, b_m, b_u)
    accelerator.backward(loss)
    
# ultra: 手动管理梯度累积
loss = loss / grad_acc_steps
loss.backward()
if acc_step == 0:
    optimizer.step()
```

## 迁移方案

### 方案A：最小侵入式（推荐）

**思路**：在ultra中添加EEG专用的data和model模块，复用ultra的分布式和checkpoint逻辑。

**目录结构**：
```
ultra/
├── ultra/
│   ├── data.py              # 现有LLM数据加载
│   ├── data_eeg.py          # 新增：EEG数据加载
│   ├── model.py             # 现有LLM模型
│   ├── model_eeg.py         # 新增：MAE模型
│   └── ...
├── main/
│   ├── train.py             # 现有LLM训练
│   └── train_eeg.py         # 新增：EEG训练
└── configs/
    ├── llm/                 # 现有LLM配置
    └── eeg/                 # 新增：EEG配置
        ├── data_eeg.yaml
        ├── model_mae.yaml
        └── train_eeg.yaml
```

**优点**：
- 不影响现有LLM训练代码
- 可以复用ultra的分布式、checkpoint、logging等基础设施
- EEG和LLM训练独立维护

**缺点**：
- 需要维护两套训练入口

### 方案B：统一接口

**思路**：抽象出通用的训练接口，让LLM和EEG共享同一个train.py。

**优点**：
- 代码更统一
- 易于添加新的任务类型

**缺点**：
- 需要较大改动现有代码
- 抽象层可能降低灵活性

## 推荐实施步骤（方案A）

### Step 1: 创建EEG数据加载模块

创建 `ultra/data_eeg.py`，包含：
1. `EEGDataArgs` - 数据配置
2. `EEGDataset` - 从eeg_infra移植的Dataset
3. `GroupedSampler` - 从eeg_infra移植
4. `build_eeg_dataloader` - 构建dataloader的函数

### Step 2: 创建EEG模型模块

创建 `ultra/model_eeg.py`，包含：
1. 从eeg_infra移植MAE、REVE、backbone
2. 适配ultra的并行化接口（FSDP wrap）

### Step 3: 创建EEG训练脚本

创建 `main/train_eeg.py`，基于 `train.py` 修改：
1. 替换数据加载为EEG
2. 替换模型为MAE
3. 调整训练循环（去掉tokenizer相关逻辑）

### Step 4: 创建配置文件

创建配置文件支持EEG训练参数。

## 关键适配点

### 1. **数据加载适配**

需要将eeg_infra的数据加载逻辑包装成ultra兼容的接口：

```python
# ultra期望的接口
def build_dataloader_from_args(args, state):
    """返回一个迭代器，每次yield (batch, new_state)"""
    pass
```

EEG数据需要：
- 支持分布式采样（DistributedSampler）
- 支持状态保存/恢复（checkpoint resume）
- 返回格式统一

### 2. **模型并行化适配**

ultra使用FSDP，需要指定哪些层wrap：

```python
# 需要为MAE模型定义FSDP wrap策略
def build_fsdp_grouping_plan_eeg(model_args):
    return {
        "encoder": ["transformer.layers"],
        "decoder": ["decoder.layers"],
    }
```

### 3. **Checkpoint适配**

ultra的CheckpointManager需要知道如何保存/加载：
- 模型state_dict
- 优化器state_dict  
- 训练state（step, acc_step, data_loader_state）

EEG的data_loader_state结构与LLM不同，需要适配。

## 下一步行动

1. 我可以先帮你创建 `ultra/data_eeg.py` 模块
2. 然后创建 `ultra/model_eeg.py` 模块
3. 最后创建 `main/train_eeg.py` 和配置文件

你想从哪个开始？或者有什么特定的需求？
