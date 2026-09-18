# Ultra训练挂死问题排查指南

## 问题现状
训练在10.60.137.131服务器上静默挂死，DataLoader单独测试正常。

## 已完成的优化
1. ✅ Positions懒加载 - 减少内存占用
2. ✅ 数据路径自动检测 - 支持不同目录结构
3. ✅ Segment过滤 - 跳过未加载的recordings

## 排查步骤（按顺序执行）

### 步骤1: 运行调试脚本（定位挂死点）
```bash
ssh 10.60.137.131
cd /mnt/zehao/ultra
bash run_debug_on_131.sh
```
**查看输出**: 找到最后完成的步骤，确定挂死位置

### 步骤2: 测试不同subset和workers配置
```bash
bash test_small_subset.sh
```
**预期结果**:
- small subset应该成功
- all subset + num_workers=0 如果成功 → 问题在workers
- all subset + num_workers=0 如果挂死 → 问题在数据量或模型

### 步骤3: 如果是workers问题
数据量太大时workers可能导致问题：
- **方案A**: 不使用workers (`num_workers=0`)
  - 优点: 稳定
  - 缺点: 数据加载慢，可能成为瓶颈
  
- **方案B**: 减少segments（限制数据量）
  修改`ultra/data_eeg.py`，在segment过滤后添加：
  ```python
  # 限制segments数量避免内存问题
  if len(self.segments) > 1000000:  # 100万
      import random
      random.shuffle(self.segments)
      self.segments = self.segments[:1000000]
      print(f"Limited segments to {len(self.segments)} for memory efficiency")
  ```

### 步骤4: 如果是forward pass问题
检查GPU内存：
```bash
# 训练启动后立即查看
nvidia-smi -l 1
```

如果OOM，减小batch size:
```bash
data.batch_size=2  # 从16/4减小到2
```

### 步骤5: 如果是分布式同步问题
可能是NCCL配置问题，尝试：
```bash
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
```

## 快速workaround（如果以上都不行）

### 方案1: 单GPU训练
```bash
export CUDA_VISIBLE_DEVICES=4
python main/train_eeg.py \
  config=configs/eeg/base.yaml \
  data.data_path=/mnt_upfs/zehao/database/egg/reve_official_data \
  data.subset=all \
  data.num_workers=0 \
  steps=1000
```

### 方案2: 使用eeg_infra框架
ultra可能有特定问题，eeg_infra框架已验证可工作：
```bash
cd /mnt/zehao/eeg_infra
# 修改数据路径到131服务器
bash test_eeg_infra.sh
```

## 调试命令速查

```bash
# 查看训练进程
ps aux | grep train_eeg | grep -v grep

# 查看GPU使用
watch -n 1 nvidia-smi

# 实时查看日志
tail -f train*.log

# 查看最后的日志
tail -100 train*.log

# 杀死卡住的进程
pkill -9 -f train_eeg
```

## 联系信息
如果问题持续，收集以下信息：
1. `bash run_debug_on_131.sh` 的完整输出
2. `test_small_subset.sh` 的结果
3. GPU内存使用情况
4. 具体挂死时的最后10行日志
