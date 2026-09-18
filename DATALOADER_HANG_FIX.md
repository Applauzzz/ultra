# DataLoader 卡死问题解决方案

## 问题描述

**现象：**
- ✅ 4 GPU + small subset → 正常
- ❌ 4 GPU + all subset + num_workers > 0 → 卡死
- ✅ 1 GPU + all subset + num_workers > 0 → 正常
- ✅ 4 GPU + all subset + num_workers=0 → **正常**

**卡死位置：**
在`iter(data_loader)`创建迭代器时，进程间死锁

## 根本原因

**DataLoader workers死锁**

当使用多GPU + 大数据集 + workers时：
- 4个GPU进程 × 2个workers = 8个并发worker进程
- 每个worker需要访问12M个segments的索引
- Workers之间+主进程之间的同步/通信出现死锁

**为什么small subset没问题：**
- small只有~5K segments，内存占用小，初始化快
- all有12M segments，worker初始化时间长，增加死锁概率

## 解决方案

### ✅ 已实施的修复

**方案A：禁用workers（推荐用于all subset）**

```bash
# test_train_4gpu.sh 已更新
python -m torch.distributed.run --nproc_per_node=4 \
  main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.num_workers=0 \  # 添加此行
  ...
```

**权衡：**
- 优点：稳定，不会死锁
- 缺点：数据加载在主进程，可能成为瓶颈
- 实测：由于GPU计算时间长，数据加载不是瓶颈

### 其他可选方案

**方案B：减少workers数量**
```bash
data.num_workers=1  # 4 GPU × 1 worker = 4个并发
```
可能工作，但仍有小概率死锁

**方案C：增大batch size**
```bash
data.batch_size=64  # 减少迭代次数，降低worker压力
```
配合num_workers=0使用可提升吞吐

## 配置建议

### Small Subset（~5K samples）
```yaml
data:
  num_workers: 2-4  # 可以使用workers
  batch_size: 16
```

### All Subset（12M samples）

**单GPU：**
```yaml
data:
  num_workers: 2-4  # 可以使用workers
  batch_size: 32
```

**4 GPU：**
```yaml
data:
  num_workers: 0     # 必须禁用workers
  batch_size: 32-64  # 增大batch补偿
```

## 技术细节

### 已完成的优化
1. ✅ Positions懒加载 - 减少内存占用
2. ✅ Segment过滤 - 只保留有效recordings
3. ✅ 数据路径自动检测
4. ✅ 禁用meta_init - 避免模型初始化卡死

### 为什么懒加载不能完全解决问题
- Positions懒加载减少了**内存占用**
- 但12M segments的**索引列表**仍需在每个worker中复制
- Workers初始化时的**进程fork**仍可能死锁

## 验证步骤

```bash
# 测试4 GPU + all subset
cd /mnt/zehao/ultra
bash test_train_4gpu.sh  # 现在应该能正常运行

# 如果需要调试
bash run_pinpoint.sh  # 精确定位卡死位置
bash run_debug_4gpu_all.sh  # 测试不同workers配置
```

## 相关文件

- `/mnt/zehao/ultra/test_train_4gpu.sh` - 已修复的4 GPU测试脚本
- `/mnt/zehao/ultra/configs/eeg/base.yaml` - base配置（默认num_workers=2）
- `/mnt/zehao/ultra/ultra/data_eeg.py` - DataLoader实现（已优化）

## 参考

- Issue: 4 GPU训练在all subset时静默卡死
- 修复: 添加`data.num_workers=0`到命令行参数
- 验证: 2026-09-10，4 GPU + all subset训练成功
