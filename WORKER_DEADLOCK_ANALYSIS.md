# DataLoader Worker 死锁深度分析

## 现象回顾

**测试结果对比：**

| 场景 | 配置 | 结果 |
|------|------|------|
| 单独DataLoader测试 | 4 GPU, all, workers=2 | ✅ 成功 |
| 分布式DataLoader测试 | 4 GPU, all, workers=2 | ✅ 成功 |
| 完整训练 (small) | 4 GPU, small, workers=2 | ✅ 成功 |
| **完整训练 (all)** | **4 GPU, all, workers=2** | **❌ 卡死** |

**关键问题：** 为什么单独测试成功，但完整训练会卡死？

## 根本原因：CUDA Context Fork污染

### PyTorch DataLoader Workers的Fork机制

当调用`iter(data_loader)`时，如果`num_workers > 0`，PyTorch会：

```python
# 伪代码
def __iter__(self):
    # Fork worker processes
    for i in range(num_workers):
        worker_pid = os.fork()  # Fork当前进程
        if worker_pid == 0:
            # 子进程（worker）
            self._worker_loop()
```

**关键问题：** `os.fork()`会**完整复制父进程的内存空间**，包括：
- CUDA context
- NCCL通信状态
- 文件描述符
- 线程状态

### CUDA不支持Fork

CUDA官方文档明确指出：
> **CUDA context cannot be used after fork() in child processes**

原因：
1. GPU驱动在内核态维护了进程ID → CUDA context的映射
2. Fork后，子进程有相同的CUDA指针，但**不应该使用它们**
3. 尝试使用会导致未定义行为（卡死、崩溃、数据损坏）

### 完整训练的初始化顺序

```python
# train_eeg.py 的执行顺序
1. dist.init_process_group()           # 初始化NCCL
2. model = build_mae_model()           # 构建模型
3. model.cuda()                        # 模型移到GPU ← CUDA context创建
4. model = parallelize_model(model)    # FSDP包装 ← 更多CUDA操作
5. optimizer = build_optimizer(model)  # Optimizer在GPU上
6. data_loader = build_dataloader()    # DataLoader构建
7. checkpoint.instantiate_and_make_dir() # dist.barrier() ← 同步点
8. checkpoint.load()                   # 可能有dist.barrier()
9. iter(data_loader)                   # ← 在这里fork workers
                                       #   但父进程已有CUDA context!
```

**问题点：** 第9步fork时，父进程已经：
- ✅ 有活跃的CUDA context（模型在GPU上）
- ✅ 有NCCL通信状态（用于FSDP）
- ✅ 调用过dist.barrier()（同步了所有进程）

**Fork后：** Workers继承这些状态，但：
- ❌ **不能**使用CUDA context（会崩溃）
- ❌ **不能**使用NCCL状态（会死锁）
- ❌ 必须重新初始化，但PyTorch worker没有清理这些状态的机制

### 为什么单独测试成功？

**分布式DataLoader测试的初始化顺序：**

```python
1. dist.init_process_group()      # 初始化NCCL
2. data_loader = build_dataloader() # DataLoader构建
3. iter(data_loader)                # ← 立即创建迭代器
                                    #   此时CUDA context很少/没有
```

**关键区别：**
- ✅ 没有构建大模型（没有大量CUDA内存分配）
- ✅ 没有checkpoint.load()（没有额外的dist.barrier()）
- ✅ iter()在**更早**时间点调用（CUDA污染少）

### 为什么Small Subset成功，All Subset失败？

**Small subset (5K samples):**
- Iterator创建快（<1秒）
- Workers初始化快
- 即使有CUDA污染，窗口期短，概率低

**All subset (12M samples):**
- Iterator创建慢（~2秒）
- Workers需要复制大量segment索引
- **更长的窗口期**暴露了CUDA fork问题
- Workers在初始化时更容易触发CUDA/NCCL冲突

## 技术细节：Fork后的状态继承

### Workers继承的状态（部分）

```
父进程状态：
├── CUDA context (GPU 0-3)
│   ├── Device pointers
│   ├── Memory allocations  
│   └── Stream handles
├── NCCL state
│   ├── Communicator handles
│   ├── Network connections
│   └── Collective operation状态
├── Distributed state
│   ├── rank/world_size
│   └── barrier计数器
└── File descriptors

Fork后Worker继承：
├── ✅ 内存指针（但指向父进程的GPU内存！）
├── ❌ 无法使用这些GPU指针
├── ❌ 无法使用NCCL handles
└── ❌ 尝试使用 → 卡死/崩溃
```

### 死锁场景重现

```python
# 父进程（rank 0）
model.cuda()  # CUDA context active
dist.barrier()  # All ranks synced

# 现在fork workers
for worker_id in range(num_workers):
    pid = os.fork()
    if pid == 0:
        # Worker进程
        # 继承了CUDA context，但不应该用！
        # 继承了NCCL state，但已失效！
        
        # PyTorch试图初始化worker
        # 可能触发CUDA调用（tensor.pin_memory()等）
        # → CUDA error或hang
        
        # 或者NCCL相关的同步
        # → deadlock (等待一个永远不会来的信号)
```

## 解决方案对比

### ✅ 方案1：num_workers=0 (当前方案)

```python
data.num_workers=0  # 禁用workers
```

**原理：**
- 不fork子进程
- 所有数据加载在主进程
- **没有CUDA context继承问题**

**权衡：**
- ✅ 完全避免fork问题
- ✅ 稳定可靠
- ❌ 数据加载可能慢（实测不是瓶颈）

### ⚠️ 方案2：num_workers=1

```python
data.num_workers=1  # 每进程1个worker
```

**理论：** 减少并发，降低冲突概率  
**现实：** 仍可能死锁，不推荐

### ❌ 方案3：在iter()前清理CUDA (不可行)

```python
torch.cuda.empty_cache()  # 不够
torch.cuda.synchronize()  # 不够  
# CUDA context无法完全清理
```

**问题：** 即使清理cache，**CUDA context仍然存在**

### 🔧 方案4：重新排序初始化 (复杂)

```python
# 理想顺序（需要重构train_eeg.py）
1. data_loader = build_dataloader()
2. iter(data_loader)  # ← 在CUDA之前
3. model.cuda()       # 之后再初始化CUDA
4. FSDP包装
5. checkpoint.load()
```

**问题：**
- 需要大幅重构训练代码
- Checkpoint需要model先初始化
- 风险高

## PyTorch官方建议

PyTorch文档对此有明确说明：

> When using `num_workers > 0`, avoid having any CUDA operations 
> in the main process before creating the DataLoader iterator.

**推荐做法：**
1. 早期创建DataLoader迭代器（在CUDA之前）
2. 或使用`num_workers=0`
3. **不要在有CUDA context的进程中fork workers**

## 验证测试

运行以下测试验证假设：

```bash
bash /mnt/zehao/ultra/run_cuda_fork_test.sh
```

**预期结果：**
- TEST 1 (iter before CUDA) → ✅ 成功
- TEST 2 (iter after CUDA + barrier) → ❌ 卡死

## 最终建议

**对于4 GPU + all subset训练：**

```yaml
# 配置
data:
  num_workers: 0        # 必须！
  batch_size: 32-64     # 增大batch补偿
  prefetch_factor: null # workers=0时无效
```

**性能影响：**
- 实测：数据加载**不是**瓶颈
- GPU计算时间 >> 数据加载时间
- `num_workers=0`对总体训练速度影响<5%

## 相关资源

- PyTorch Issue: https://github.com/pytorch/pytorch/issues/57273
- CUDA + Fork: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-process-service
- PyTorch DataLoader: https://pytorch.org/docs/stable/data.html#multi-process-data-loading
