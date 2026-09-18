#!/usr/bin/env python3
"""Pinpoint exact location of hang in 4 GPU + all config."""

import os
import sys
import time
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Setup timeout handler
def timeout_handler(signum, frame):
    rank = int(os.environ.get("RANK", 0))
    print(f"[RANK {rank}] TIMEOUT! Hung at this step.", flush=True)
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)

def log(msg):
    rank = int(os.environ.get("RANK", 0))
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [RANK {rank}] {msg}", flush=True)

# Step 1: Imports
log("STEP 1: Importing...")
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader
import torch
import torch.distributed as dist
log("✓ STEP 1 DONE")

# Step 2: Distributed init
log("STEP 2: Init distributed...")
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
if world_size > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
log("✓ STEP 2 DONE")

# Step 3: Build dataloader
log("STEP 3: Building DataLoader (all subset, 2 workers)...")
signal.alarm(120)  # 2 min timeout for this step

data_args = EEGDataArgs(
    data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
    subset="all",
    batch_size=16,
    num_workers=2,
    persistent_workers=False,
    world_size=world_size,
    rank=rank,
)
data_loader, _ = build_eeg_dataloader(data_args, train=True)
signal.alarm(0)
log(f"✓ STEP 3 DONE ({len(data_loader)} batches)")

# Step 4: Create iterator - THIS IS LIKELY WHERE IT HANGS
log("STEP 4: Creating iterator...")
log("  Calling iter(data_loader)...")
signal.alarm(180)  # 3 min timeout

iter_start = time.time()
data_iter = iter(data_loader)
iter_time = time.time() - iter_start

signal.alarm(0)
log(f"✓ STEP 4 DONE (took {iter_time:.1f}s)")

# Step 5: Load first batch
log("STEP 5: Loading first batch...")
signal.alarm(60)

batch = next(data_iter)

signal.alarm(0)
log("✓ STEP 5 DONE")

# Step 6: Barrier sync
if world_size > 1:
    log("STEP 6: Barrier sync...")
    dist.barrier()
    log("✓ STEP 6 DONE")

log("="*60)
log("ALL STEPS COMPLETED - NO HANG!")
log("="*60)
