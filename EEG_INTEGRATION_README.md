# EEG训练Pipeline集成到Ultra框架

## 已完成的工作

### 1. 创建了EEG数据加载模块 (`ultra/data_eeg.py`)

这个模块包含：
- ✅ `EEGDataArgs` - 数据配置dataclass
- ✅ `EEGDataset` - 从eeg_infra移植的Dataset（支持memmap加载）
- ✅ `GroupedSampler` - 按通道数分组的采样器（保证batch内通道数一致）
- ✅ `build_eeg_dataloader()` - 构建dataloader的函数
- ✅ `EEGDataLoaderState` - 用于checkpoint resume的状态管理

**关键特性：**
- 完全兼容ultra的分布式训练（world_size, rank参数）
- 支持状态保存/恢复（可以从checkpoint resume）
- 使用memmap零拷贝加载大规模EEG数据
- GroupedSampler确保IO局部性（减少随机读取）
- 支持persistent_workers（推荐设为True）

## 下一步需要做的事情

### Step 1: 创建EEG模型模块（推荐下一步）

需要创建 `ultra/model_eeg.py`，包含：

```python
# 需要从eeg_infra移植的模块
from eeg_infra.src.models.mae import MAE
from eeg_infra.src.models.encoder import REVE, FourierEmb4D
from eeg_infra.src.models.backbone import Transformer, FlashAttention

# 适配ultra的FSDP
@dataclass
class MAEModelArgs:
    encoder_config: str = "base"  # base, large, small, tiny
    decoder_config: str = "base"
    masking_ratio: float = 0.75
    patch_size: int = 200
    patch_overlap: int = 20
    # ... 其他参数

def build_mae_model(args: MAEModelArgs) -> MAE:
    """构建MAE模型，兼容ultra的并行化"""
    pass

def build_fsdp_grouping_plan_eeg(args: MAEModelArgs):
    """定义FSDP wrap策略"""
    return {
        "encoder.transformer.layers": TransformerBlock,
        "decoder.layers": TransformerBlock,
    }
```

### Step 2: 创建EEG训练脚本

创建 `main/train_eeg.py`，基于`main/train.py`修改：

**主要修改点：**

```python
# 1. 导入EEG模块
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader, EEGDataLoaderState
from ultra.model_eeg import MAEModelArgs, build_mae_model, build_fsdp_grouping_plan_eeg

# 2. 修改TrainArgs
@dataclass
class EEGTrainArgs:
    name: str = "eeg_mae"
    dump_dir: str = ""
    seed: int = 42
    grad_acc_steps: int = 1
    steps: int = 1000
    
    data: EEGDataArgs = field(default_factory=EEGDataArgs)  # ← EEG数据配置
    model: MAEModelArgs = field(default_factory=MAEModelArgs)  # ← MAE模型配置
    optim: OptimArgs = field(default_factory=OptimArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

# 3. 修改训练循环
def train(args: EEGTrainArgs):
    # ... setup ...
    
    # 不需要tokenizer
    # tokenizer = build_tokenizer(...)  # ← 删除
    
    # 构建MAE模型
    model = build_mae_model(args.model)
    
    # 并行化模型
    model = parallelize_model(
        model,
        world_mesh,
        args.model,
        args.distributed,
        fsdp_grouping_plan=build_fsdp_grouping_plan_eeg(args.model),  # ← EEG的wrap策略
        tp_parallelize=None,
        no_recompute_ops=None,
    )
    
    # 构建EEG dataloader
    data_loader, data_loader_state = build_eeg_dataloader(
        args.data,
        train=True,
        state=None,  # 或从checkpoint恢复
    )
    
    # 训练循环
    for eeg, pos, batch_mask, batch_unmask in data_loader:
        # EEG forward
        loss = model(eeg, pos, batch_mask, batch_unmask)
        
        loss = loss / args.grad_acc_steps
        loss.backward()
        
        if acc_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

### Step 3: 创建配置文件

创建 `configs/eeg/train_eeg_base.yaml`:

```yaml
# EEG Training Configuration
name: eeg_mae_base
dump_dir: /path/to/outputs/eeg_mae

seed: 42
steps: 10000
grad_acc_steps: 1

data:
  data_path: /mnt/zehao/eeg_infra/data
  subset: all
  window_duration: 10000
  clip: 10.0
  
  # Masking
  masking_ratio: 0.75
  masking_window: 200
  masking_overlap: 20
  use_block_masking: false
  
  # DataLoader
  batch_size: 448
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true

model:
  encoder_config: base  # 512 dim, 22 layers
  decoder_config: base
  patch_size: 200
  patch_overlap: 20
  freqs: 4
  noise_ratio: 0.0025

optim:
  lr: 2.4e-4
  betas: [0.9, 0.95]
  weight_decay: 0.0
  clip: 5.0
  scheduler: cosine  # or trapezoid

distributed:
  dp_replicate: 1
  dp_shard: 4  # FSDP over 4 GPUs
  tp_size: 1
  sp_size: 1

checkpoint:
  dump:
    every: 1000
  keep_latest: 3

logging:
  freq: 10
  wandb:
    project: eeg_mae
    entity: your_entity
```

### Step 4: 运行训练

```bash
# 4 GPU训练
cd /mnt/zehao/ultra

torchrun --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/train_eeg_base.yaml \
  data.data_path=/mnt/zehao/eeg_infra/data \
  name=eeg_base_test
```

## 与原eeg_infra的对比

| 特性 | eeg_infra | ultra集成 |
|------|-----------|----------|
| **分布式** | Accelerate (DDP) | PyTorch FSDP + TP + SP |
| **配置** | Hydra | Hydra (相同) |
| **数据加载** | ✅ 已移植 | ✅ `ultra/data_eeg.py` |
| **模型** | MAE | ⏳ 待移植到 `ultra/model_eeg.py` |
| **训练循环** | Accelerate管理 | ⏳ 待创建 `main/train_eeg.py` |
| **Checkpoint** | Accelerate | Ultra CheckpointManager |
| **Logging** | tqdm + wandb | Ultra MetricLogger + wandb |

## 优势

使用ultra框架后，你可以：

1. ✅ **更强的并行化能力**
   - FSDP: 将模型参数分片到多卡（支持超大模型）
   - TP: Tensor Parallelism（可选）
   - SP: Sequence Parallelism（可选，对长序列EEG有用）

2. ✅ **统一的训练基础设施**
   - 与LLM训练共享checkpoint、logging、distributed逻辑
   - 更好的代码复用

3. ✅ **更灵活的配置系统**
   - 继承ultra的完整配置体系
   - 支持配置组合和覆盖

4. ✅ **生产级checkpoint管理**
   - 自动保存/恢复
   - 分布式checkpoint（每个rank只保存自己的shard）

## 注意事项

### 1. DataLoader与ultra现有data.py的区别

- **ultra现有**: JSONL文本 → tokenize → pack tokens → prefetch
- **EEG新增**: Memmap → window sampling → normalization → masking

两者是**并列关系**，不是替换。`ultra/data.py`继续服务LLM训练，`ultra/data_eeg.py`服务EEG训练。

### 2. 是否需要修改ultra核心代码？

**不需要！** 这就是方案A（最小侵入式）的优势：
- `ultra/data_eeg.py` - 新增文件
- `ultra/model_eeg.py` - 新增文件（待创建）
- `main/train_eeg.py` - 新增文件（待创建）
- 其他ultra核心代码 - **完全不改动**

### 3. Checkpoint兼容性

EEG的checkpoint与LLM checkpoint格式不同：
- **LLM**: `{model, optimizer, train_state: {step, data_loader_state: PackTokensState}}`
- **EEG**: `{model, optimizer, train_state: {step, data_loader_state: EEGDataLoaderState}}`

Ultra的`CheckpointManager`已经支持这种灵活性，只需确保`EEGDataLoaderState`实现`state_dict()`和`load_state_dict()`即可（已在`data_eeg.py`中实现）。

## 快速开始（Demo）

如果你想先测试数据加载是否工作：

```python
# test_eeg_data.py
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

# 配置
args = EEGDataArgs(
    data_path="/mnt/zehao/eeg_infra/data",
    subset="all",
    batch_size=448,
    num_workers=4,
    world_size=1,
    rank=0,
)

# 构建dataloader
dataloader, state = build_eeg_dataloader(args, train=True)

# 测试加载
for batch_idx, (eeg, pos, mask, unmask) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  EEG shape: {eeg.shape}")  # 应该是 (batch_size, channels, time)
    print(f"  Positions shape: {pos.shape}")  # (batch_size, channels, 3)
    print(f"  Mask indices: {mask.shape}")
    print(f"  Unmask indices: {unmask.shape}")
    
    if batch_idx >= 2:
        break

print("\n✅ EEG数据加载测试成功！")
```

运行测试：
```bash
cd /mnt/zehao/ultra
python test_eeg_data.py
```

## 我可以帮你做什么？

1. **创建 `ultra/model_eeg.py`** - 移植MAE模型并适配FSDP
2. **创建 `main/train_eeg.py`** - 完整的EEG训练脚本
3. **创建配置文件** - `configs/eeg/` 目录下的YAML配置
4. **测试脚本** - 确保数据加载和模型训练工作正常

你想让我先做哪一个？
