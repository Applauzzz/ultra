# Worker死锁的根本原因：初始化顺序

## 关键发现

**不是NCCL环境变量**，而是**DataLoader创建时机 vs 分布式初始化顺序**

## 两个框架的初始化顺序对比

### eeg_infra (✅ workers正常)

```python
# src/train.py

# 1. 创建accelerator（但还没初始化NCCL）
accelerator = get_accelerator(args)

# 2. 创建模型（普通模型，还不是分布式）
mae = MAE(args)

# 3. 【关键】创建DataLoader - 此时NCCL还没初始化！
train_loader, _, len_train, ... = get_train_val_loaders(args, return_val=False)

# 4. 准备模型 - 【这时候才初始化NCCL】
mae = accelerator.prepare(mae)

# 5. 准备optimizer
optimizer = get_optimizer(...)
optimizer = accelerator.prepare(optimizer)

# 6. 准备DataLoader - accelerate内部处理了workers
train_loader = accelerator.prepare(train_loader)

# 7. 开始训练
for epoch in range(...):
    for batch in train_loader:  # ← 已经在accelerator.prepare()中处理好了
        ...
```

**时间线**：
```
创建DataLoader → accelerator.prepare(model) → 初始化NCCL → accelerator.prepare(dataloader) → iter()
     ↑                                               ↑                                            ↑
  此时还是普通对象                                NCCL初始化                               Workers已被accelerate处理
```

### ultra (❌ workers死锁)

```python
# main/train_eeg.py

# 1. 【关键】先初始化分布式 - NCCL立即初始化！
setup_torch_distributed(args.distributed)
world_mesh = get_device_mesh(args.distributed)

# 2. 构建模型
model = build_mae_model(args.model)

# 3. 并行化模型 - 进一步初始化NCCL通信通道
model = parallelize_model(model, world_mesh, ...)  # FSDP初始化

# 4. 构建optimizer
optimizer, scheduler = build_optimizer(model, ...)

# 5. 【关键】最后才创建DataLoader - 但NCCL已经初始化了！
data_loader, data_loader_state = build_eeg_dataloader(args.data, ...)

# 6. 转为iterator - 这里fork创建workers
data_loader = iter(data_loader)  # ← Fork时继承了NCCL状态！

# 7. 开始训练
while train_state.step < args.steps:
    eeg, pos, batch_mask, batch_unmask = next(data_loader)  # ← 死锁！
    ...
```

**时间线**：
```
初始化NCCL → parallelize_model → 更多NCCL初始化 → 创建DataLoader → iter() → Fork workers
     ↑                                   ↑                                  ↑            ↑
  打开网络设备                    建立通信通道                        主进程已有NCCL状态   继承NCCL状态！
```

## 为什么会死锁

### NCCL初始化会做什么

```c
// NCCL内部 (C++)
ncclInit() {
    // 1. 打开网络设备（InfiniBand / ethernet）
    int fd = open("/dev/infiniband/uverbs0", O_RDWR);  // 文件描述符
    
    // 2. 创建socket连接
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    bind(sock, interface_eth1, ...);  // 绑定到eth1接口
    
    // 3. 建立ring/tree通信拓扑
    connect_to_other_ranks(...);
    
    // 4. 分配GPU内存和通信缓冲区
    cudaMalloc(&send_buffer, ...);
    cudaMalloc(&recv_buffer, ...);
    
    // 5. 注册内存到网络设备（RDMA）
    ibv_reg_mr(fd, send_buffer, ...);
}
```

### Fork时发生了什么

```
父进程 (Rank 0):
├── NCCL state:
│   ├── /dev/infiniband/uverbs0 fd=5  ← 文件描述符
│   ├── socket fd=6 (bound to eth1)   ← socket
│   ├── send_buffer GPU pointer       ← GPU内存
│   └── recv_buffer GPU pointer
└── PyTorch DataLoader

        ↓ os.fork() × 2 (num_workers=2)

Worker 1 (PID=12345):          Worker 2 (PID=12346):
├── 继承的NCCL state:           ├── 继承的NCCL state:
│   ├── fd=5 (same!)           │   ├── fd=5 (same!)
│   ├── fd=6 (same!)           │   ├── fd=6 (same!)
│   ├── send_buffer (invalid!)  │   ├── send_buffer (invalid!)
│   └── recv_buffer (invalid!)  │   └── recv_buffer (invalid!)
```

### 为什么死锁

1. **文件描述符冲突**
   ```
   父进程：read(fd=5) → InfiniBand设备
   Worker 1：read(fd=5) → 同一个设备！
   Worker 2：read(fd=5) → 同一个设备！
   
   设备不知道该把数据发给谁 → 数据包丢失/混乱
   ```

2. **Socket端口冲突**
   ```
   父进程：已绑定 eth1:12345
   Worker 1：继承了绑定，尝试使用 → "Address already in use"
   Worker 2：同样的问题
   
   Workers卡在等待网络初始化
   ```

3. **GPU内存指针无效**
   ```
   父进程：send_buffer = 0x7f8a00000000 (valid)
   Worker 1：send_buffer = 0x7f8a00000000 (invalid! 这是父进程的地址空间)
   
   Workers访问无效内存 → segfault 或 hang
   ```

4. **NCCL内部锁**
   ```
   父进程：持有 nccl_comm_lock
   Fork时：Worker继承锁状态（可能是locked）
   Worker：等待锁释放 → 永远等待（父进程在另一个地址空间）
   ```

## 为什么Accelerate能工作

Accelerate的聪明设计：

```python
# accelerate内部

def prepare(self, *objects):
    for obj in objects:
        if isinstance(obj, DataLoader):
            # 1. 提取DataLoader的配置
            dataset = obj.dataset
            batch_size = obj.batch_size
            num_workers = obj.num_workers
            
            # 2. 【关键】在每个进程中重新创建DataLoader
            # 而不是直接使用传入的DataLoader
            new_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                # 每个进程独立创建，避免继承问题
            )
            
            return new_loader
```

**为什么有效**：
- DataLoader在每个已经初始化好NCCL的进程中**独立创建**
- 每个进程的workers fork时，继承的是**本进程的**NCCL状态（一致的）
- 不会跨进程继承

## 解决方案

### 方案1：模仿eeg_infra - 调整初始化顺序 (复杂)

```python
# 在 train_eeg.py 中

def train(args):
    # 1. 先创建DataLoader（分布式初始化之前）
    data_loader, data_loader_state = build_eeg_dataloader(
        args.data, train=True, state=None
    )
    
    # 2. 然后初始化分布式
    setup_torch_distributed(args.distributed)
    
    # 3. 但是！DataLoader不知道自己的rank...
    # 这需要大量重构
```

**问题**：
- DataLoader需要rank信息来shard数据
- 但rank信息在分布式初始化后才有
- 需要大量架构改动

### 方案2：使用spawn而不是fork (PyTorch限制)

```python
# torch.multiprocessing
mp.set_start_method('spawn')  # 而不是'fork'
```

**问题**：
- PyTorch DataLoader只支持fork，不支持spawn
- 这不是我们能改的

### 方案3：num_workers=0 (当前方案，最简单)

```bash
data.num_workers=0
```

**优点**：
- ✅ 零代码改动
- ✅ 完全避免fork
- ✅ 稳定可靠

**缺点**：
- ❌ 性能损失 ~5%（但GPU是瓶颈，实际影响小）

### 方案4：切换到Accelerate (大改，最优)

重写训练代码使用accelerate，像eeg_infra一样。

**优点**：
- ✅ 可以使用workers
- ✅ 更高级的抽象
- ✅ 更好的容错性

**缺点**：
- ❌ 需要重写整个训练循环
- ❌ 工作量大（1-2天）

## 总结

| 方面 | eeg_infra | ultra |
|------|-----------|-------|
| 框架 | Accelerate | PyTorch Distributed |
| DataLoader创建时机 | **NCCL初始化之前** | **NCCL初始化之后** |
| Workers | ✅ 正常工作 | ❌ 继承NCCL状态 → 死锁 |
| 解决方案 | 设计已避免问题 | 需要num_workers=0 |

**根本原因**：
- ❌ **不是**NCCL环境变量
- ❌ **不是**CUDA context
- ✅ **是**初始化顺序：ultra在NCCL初始化**之后**创建DataLoader并fork

**推荐方案**：
- 短期：`num_workers=0`（性能损失小，稳定）
- 长期：考虑迁移到Accelerate框架（如果需要workers性能）
