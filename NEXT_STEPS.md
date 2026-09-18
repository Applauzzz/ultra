# 诊断测试 - 下一步

## 当前状态

✅ **问题已解决**：`data.num_workers=0` 让训练正常运行

❌ **CUDA假设被推翻**：test_cuda_fork_issue.py测试通过，CUDA context不是根因

## 待运行的3个诊断测试

这些测试帮助理解**为什么**workers会死锁（纯学术兴趣）：

### 1. NCCL环境变量测试

```bash
ssh 10.60.137.131
cd /mnt/zehao/ultra
bash test_nccl_env_fork.sh
```

**测试假设**：
- NCCL_IB_HCA=mlx5_0 (InfiniBand)
- NCCL_SOCKET_IFNAME=eth1
- 这些网络设置在fork时导致冲突

**预期结果**：
- 没有NCCL环境变量 → 快速成功
- 有NCCL环境变量 → 变慢或卡死

### 2. WandB线程测试

```bash
ssh 10.60.137.131
cd /mnt/zehao/ultra
bash run_wandb_test.sh
```

**测试假设**：
WandB的后台线程（文件监控、指标缓冲）在fork时导致死锁

**预期结果**：
- 如果初始化WandB后卡死 → WandB线程是根因

### 3. 完整序列测试

```bash
ssh 10.60.137.131
cd /mnt/zehao/ultra
bash run_full_sequence_test.sh
```

**测试假设**：
不是单一因素，而是完整训练环境的组合：
- 模型 (95M) + FSDP + NCCL + WandB + 大数据集

**预期结果**：
精确定位死锁发生在哪一步

## 关键发现：环境变量差异

**成功的测试**：
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 仅此而已
```

**失败的训练**：
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_IB_HCA=mlx5_0          # ← 可疑！
export NCCL_SOCKET_IFNAME=eth1     # ← 可疑！
export GLOO_SOCKET_IFNAME=eth1     # ← 可疑！
export CUDA_LAUNCH_BLOCKING=1
export TRITON_AUTOTUNE=0
```

**最可能的根因**：
NCCL网络设置 + fork → 网络状态冲突

## 下一步行动

**选项A（推荐）**：问题已解决，无需进一步诊断
- `num_workers=0`稳定有效
- 性能影响 <5%
- 可以开始正式训练

**选项B（学术兴趣）**：运行3个测试找出确切原因
- 帮助理解PyTorch + NCCL + fork的交互
- 可能发现PyTorch bug
- 对社区有贡献价值

你的选择？
