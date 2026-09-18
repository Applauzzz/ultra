# eeg_infra vs ultra - 为什么一个能用workers一个不能

## 关键差异：环境变量

### eeg_infra (✅ workers正常工作)

```bash
# run_pretrain_4gpu_wandb.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export SCRATCH="$OUTPUT_ROOT"
export TMPDIR="${TMPDIR:-/tmp/reve_infra_${USER:-user}}"

# 启动方式
accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision fp16 \
  src/train.py
```

**没有**设置任何NCCL网络环境变量！

### ultra (❌ workers死锁)

```bash
# test_train_4gpu.sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
export NCCL_IB_HCA=mlx5_0          # ← 问题根源！
export NCCL_SOCKET_IFNAME=eth1     # ← 问题根源！
export GLOO_SOCKET_IFNAME=eth1     # ← 问题根源！

# 启动方式
python -m torch.distributed.run \
  --nproc_per_node=4 \
  main/train_eeg.py
```

**有**NCCL网络环境变量！

## 为什么NCCL环境变量导致问题

### NCCL初始化顺序

**eeg_infra流程**：
```
1. Fork创建workers (数据加载)
   ├── Worker 1 进程
   ├── Worker 2 进程
   └── ...
   
2. NCCL初始化 (accelerate内部)
   ├── 每个rank独立初始化
   └── 不依赖环境变量中的硬件配置
   
3. 训练开始
```

**ultra流程**：
```
1. 环境变量设置了NCCL硬件配置
   export NCCL_IB_HCA=mlx5_0        # 指定InfiniBand设备
   export NCCL_SOCKET_IFNAME=eth1   # 指定网络接口
   
2. NCCL在主进程初始化
   ├── 打开InfiniBand设备 mlx5_0
   ├── 绑定网络接口 eth1
   └── 建立socket连接
   
3. Fork创建workers ← 问题发生！
   ├── Worker继承InfiniBand句柄
   ├── Worker继承socket文件描述符
   └── Worker继承网络接口绑定
   
4. Workers尝试使用继承的网络资源
   └── 冲突！多个进程共享同一个网络状态 → 死锁
```

### Fork继承的危险状态

当fork发生时，workers继承：

| 资源类型 | 继承的状态 | 为什么危险 |
|---------|----------|----------|
| **InfiniBand设备** | mlx5_0的文件描述符 | 设备不支持多进程同时访问同一个句柄 |
| **Socket连接** | eth1上已建立的socket | 内核不知道该把数据包发给父进程还是worker |
| **网络缓冲区** | NCCL的发送/接收缓冲区 | 缓冲区指针在worker中无效 |
| **NCCL通信上下文** | 集合通信的状态机 | 状态机在fork后不一致 |

## 解决方案对比

### 方案1：移除NCCL环境变量 (推荐)

```bash
# 修改 test_train_4gpu.sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 删除这些：
# export NCCL_IB_HCA=mlx5_0
# export NCCL_SOCKET_IFNAME=eth1
# export GLOO_SOCKET_IFNAME=eth1
```

**优点**：
- ✅ 可以使用workers
- ✅ 性能提升（workers并行加载）
- ✅ 与eeg_infra一致

**缺点**：
- ❓ NCCL会自动选择网络设备（可能不是最优）
- ❓ 需要测试网络性能是否受影响

### 方案2：使用num_workers=0 (当前方案)

```bash
data.num_workers=0
```

**优点**：
- ✅ 稳定可靠
- ✅ 避免所有fork相关问题

**缺点**：
- ❌ 性能损失~5%（单进程加载数据）

### 方案3：切换到accelerate (重大改动)

像eeg_infra一样使用accelerate而不是torch.distributed.run

**优点**：
- ✅ accelerate处理了这些细节
- ✅ 更高级的抽象

**缺点**：
- ❌ 需要重写训练代码
- ❌ 工作量大

## 推荐行动

### 快速验证（5分钟）

测试移除NCCL环境变量是否解决问题：

```bash
# 创建测试脚本
cat > /mnt/zehao/ultra/test_without_nccl.sh << 'SCRIPT'
#!/bin/bash
source /mnt_upfs/miniconda/etc/profile.d/conda.sh
conda activate ultra_2

export CUDA_VISIBLE_DEVICES=4,5,6,7
# 不设置NCCL环境变量

cd /mnt/zehao/ultra
python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.num_workers=2 \
  grad_acc_steps=2 \
  steps=10
SCRIPT

chmod +x /mnt/zehao/ultra/test_without_nccl.sh
ssh 10.60.137.131 'bash test_without_nccl.sh'
```

### 预期结果

如果移除NCCL环境变量后：
- ✅ 训练正常启动 → NCCL环境变量是罪魁祸首
- ❌ 仍然死锁 → 还有其他因素

## 为什么这些环境变量存在

这些NCCL变量通常用于：

1. **NCCL_IB_HCA=mlx5_0**
   - 指定使用特定的InfiniBand设备
   - 在有多个IB设备的服务器上选择最快的

2. **NCCL_SOCKET_IFNAME=eth1**
   - 指定TCP fallback使用的网络接口
   - 避免NCCL选择错误的网络（如管理网口）

3. **GLOO_SOCKET_IFNAME=eth1**
   - Gloo backend（CPU通信）使用的接口

**问题**：这些设置在**NCCL初始化之前**就绑定了硬件，导致fork时继承了绑定状态。

## 总结

| 框架 | NCCL环境变量 | Workers | 结果 |
|------|-------------|---------|------|
| eeg_infra | ❌ 无 | ✅ 16 workers | ✅ 正常 |
| ultra | ✅ 有 | ✅ 2 workers | ❌ 死锁 |
| ultra | ✅ 有 | ❌ 0 workers | ✅ 正常 |
| ultra | ❌ 无 | ❓ 2 workers | ❓ **待测试** |

**下一步**：运行 `test_without_nccl.sh` 验证假设
