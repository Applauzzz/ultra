#!/usr/bin/env python3
"""Test if CUDA context + dist.barrier() before iter() causes workers to hang."""

import os
import sys
import time
import torch
import torch.distributed as dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def log(msg):
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Init distributed
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if world_size > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

log("Distributed initialized")

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

# Test 1: iter() BEFORE any CUDA operations
log("\n" + "="*60)
log("TEST 1: Create iterator BEFORE CUDA operations")
log("="*60)

data_args = EEGDataArgs(
    data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
    subset="all",
    batch_size=16,
    num_workers=2,
    persistent_workers=False,
    world_size=world_size,
    rank=rank,
)

log("Building DataLoader...")
data_loader1, _ = build_eeg_dataloader(data_args, train=True)
log("DataLoader built")

log("Creating iterator BEFORE CUDA...")
start = time.time()
data_iter1 = iter(data_loader1)
log(f"✓ Iterator created in {time.time()-start:.1f}s (BEFORE CUDA)")

# Now do CUDA operations
log("Now initializing CUDA...")
dummy_tensor = torch.randn(10, 10).cuda()
log("CUDA initialized")

# Load a batch
batch = next(data_iter1)
log("✓ Batch loaded successfully")

# Cleanup
del data_loader1, data_iter1, batch
torch.cuda.empty_cache()
if world_size > 1:
    dist.barrier()

# Test 2: iter() AFTER CUDA operations and barrier
log("\n" + "="*60)
log("TEST 2: Create iterator AFTER CUDA + dist.barrier()")
log("="*60)

log("Initializing CUDA first...")
dummy_model = torch.nn.Linear(10, 10).cuda()
log("CUDA initialized")

if world_size > 1:
    log("Calling dist.barrier()...")
    dist.barrier()
    log("dist.barrier() completed")

log("Building DataLoader...")
data_loader2, _ = build_eeg_dataloader(data_args, train=True)
log("DataLoader built")

log("Creating iterator AFTER CUDA + barrier...")
log("  This is where training hangs with workers > 0...")
start = time.time()

import signal
signal.alarm(60)  # 1 min timeout

try:
    data_iter2 = iter(data_loader2)
    signal.alarm(0)
    log(f"✓ Iterator created in {time.time()-start:.1f}s (AFTER CUDA)")
    
    batch = next(data_iter2)
    log("✓ Batch loaded successfully")
    
except Exception as e:
    signal.alarm(0)
    log(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

log("\n" + "="*60)
log("TEST COMPLETED")
log("="*60)
