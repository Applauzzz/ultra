#!/bin/bash
echo "Comparing environment variables between test scripts and training scripts"
echo ""

echo "===== test_train_4gpu.sh environment ====="
grep "^export" /mnt/zehao/ultra/test_train_4gpu.sh | head -20

echo ""
echo "===== distributed_test script environment ====="
grep "^export" /mnt/zehao/ultra/run_distributed_test.sh 2>/dev/null || echo "(no exports found)"

echo ""
echo "===== test_dataloader_only environment ====="
grep "export" /mnt/zehao/ultra/run_dataloader_test.sh 2>/dev/null | head -5

echo ""
echo "===== Key differences ====="
echo "Training scripts may have:"
echo "  - NCCL_* environment variables"
echo "  - TRITON_* settings"
echo "  - CUDA_LAUNCH_BLOCKING"
echo ""
echo "These could interact with worker fork!"
