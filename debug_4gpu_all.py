#!/usr/bin/env python3
"""Debug 4 GPU + all subset hang issue."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def log(msg):
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

# Import and setup
log("Importing modules...")
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

log("Initializing distributed...")
import torch.distributed as dist
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
if world_size > 1:
    dist.init_process_group(backend="nccl")
    import torch
    torch.cuda.set_device(local_rank)
log(f"Distributed initialized: rank {rank}/{world_size}")

# Build DataLoader with different worker configs
for num_workers in [0, 1, 2]:
    log(f"\n{'='*60}")
    log(f"Testing with num_workers={num_workers}")
    log(f"{'='*60}")
    
    data_args = EEGDataArgs(
        data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
        subset="all",
        batch_size=16,
        num_workers=num_workers,
        persistent_workers=False,
        world_size=world_size,
        rank=rank,
    )
    
    log(f"  Building dataloader...")
    start = time.time()
    data_loader, _ = build_eeg_dataloader(data_args, train=True)
    build_time = time.time() - start
    log(f"  ✓ Built in {build_time:.1f}s: {len(data_loader)} batches")
    
    log(f"  Creating iterator...")
    iter_start = time.time()
    try:
        data_iter = iter(data_loader)
        iter_time = time.time() - iter_start
        log(f"  ✓ Iterator created in {iter_time:.1f}s")
        
        log(f"  Loading first batch...")
        batch_start = time.time()
        batch = next(data_iter)
        batch_time = time.time() - batch_start
        log(f"  ✓ First batch loaded in {batch_time:.1f}s")
        
        log(f"  Loading second batch...")
        batch2_start = time.time()
        batch2 = next(data_iter)
        batch2_time = time.time() - batch2_start
        log(f"  ✓ Second batch loaded in {batch2_time:.1f}s")
        
        log(f"  SUCCESS with num_workers={num_workers}")
        
    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        break

log("\n" + "="*60)
log("All tests completed!")
log("="*60)
