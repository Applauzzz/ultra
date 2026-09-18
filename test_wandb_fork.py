#!/usr/bin/env python3
"""Test if WandB + workers causes deadlock."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def log(msg):
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Init distributed
import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if world_size > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

log("Distributed initialized")

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

data_args = EEGDataArgs(
    data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
    subset="all",
    batch_size=16,
    num_workers=2,
    persistent_workers=False,
    world_size=world_size,
    rank=rank,
)

# Test 1: Without WandB
log("\n" + "="*60)
log("TEST 1: Create iterator WITHOUT WandB")
log("="*60)

log("Building DataLoader...")
data_loader1, _ = build_eeg_dataloader(data_args, train=True)

log("Creating iterator...")
start = time.time()
data_iter1 = iter(data_loader1)
log(f"✓ Iterator created in {time.time()-start:.1f}s")

batch = next(data_iter1)
log("✓ Batch loaded")

del data_loader1, data_iter1, batch

# Test 2: WITH WandB
log("\n" + "="*60)
log("TEST 2: Create iterator WITH WandB initialized")
log("="*60)

if rank == 0:
    log("Initializing WandB...")
    import wandb
    wandb.init(
        project="eeg",
        entity="XLM",
        mode="offline",  # offline mode to avoid network
        name="test_wandb_fork",
    )
    log("WandB initialized")
    
    # WandB starts background threads for:
    # - File watching
    # - Metric buffering
    # - System monitoring
    
    import threading
    log(f"Active threads after WandB: {threading.active_count()}")
    for t in threading.enumerate():
        log(f"  Thread: {t.name}")

if world_size > 1:
    dist.barrier()

log("Building DataLoader...")
data_loader2, _ = build_eeg_dataloader(data_args, train=True)

log("Creating iterator AFTER WandB...")
log("  If this hangs, WandB threads conflict with workers fork...")

import signal
signal.alarm(90)  # 90 sec timeout

try:
    start = time.time()
    data_iter2 = iter(data_loader2)
    signal.alarm(0)
    log(f"✓ Iterator created in {time.time()-start:.1f}s")
    
    batch = next(data_iter2)
    log("✓ Batch loaded")
    
    log("\n" + "="*60)
    log("SUCCESS - WandB does NOT cause worker deadlock")
    log("="*60)
    
except Exception as e:
    signal.alarm(0)
    log(f"\n✗ FAILED with WandB: {e}")
    import traceback
    traceback.print_exc()
    
    log("\n" + "="*60)
    log("CONFIRMED - WandB threads cause worker deadlock!")
    log("="*60)

if rank == 0 and 'wandb' in dir():
    wandb.finish()
