#!/usr/bin/env python3
"""Replicate exact training initialization sequence to find deadlock point."""

import os
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def log(msg):
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Step 1: Distributed init (same as training)
import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

log("STEP 1: Init distributed")
if world_size > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
log("✓ STEP 1")

# Step 2: Build model (big, like training)
log("STEP 2: Build large model")
from ultra.model_eeg import MAEModelArgs, MAE

model_args = MAEModelArgs()
model = MAE(model_args).cuda()
log(f"✓ STEP 2 - Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Step 3: DDP/FSDP wrapping
log("STEP 3: DDP wrapping")
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
)
log("✓ STEP 3")

# Step 4: Optimizer (same as training)
log("STEP 4: Build optimizer")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
log("✓ STEP 4")

# Step 5: Build DataLoader (with training batch size!)
log("STEP 5: Build DataLoader with LARGE batch")
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

data_args = EEGDataArgs(
    data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
    subset="all",
    batch_size=300,  # Same as training config!
    num_workers=2,
    persistent_workers=False,
    world_size=world_size,
    rank=rank,
)
data_loader, _ = build_eeg_dataloader(data_args, train=True)
log(f"✓ STEP 5 - {len(data_loader)} batches")

# Step 6: Checkpoint directory (with barrier!)
log("STEP 6: Create checkpoint dir + barrier")
if rank == 0:
    os.makedirs("/tmp/test_ckpt", exist_ok=True)
dist.barrier()
log("✓ STEP 6")

# Step 7: Initialize WandB (only rank 0, like training)
if rank == 0:
    log("STEP 7: Initialize WandB")
    import wandb
    wandb.init(
        project="eeg",
        entity="XLM",
        mode="offline",
        name="test_full_sequence",
    )
    log("✓ STEP 7")
else:
    log("STEP 7: (skip WandB on non-master)")

# Barrier after WandB
if world_size > 1:
    dist.barrier()

# Step 8: gc.disable() (like training)
log("STEP 8: Disable GC")
gc.disable()
log("✓ STEP 8")

# Step 9: Create iterator (THE CRITICAL STEP)
log("STEP 9: Create DataLoader iterator")
log("  This is where training hangs...")

import signal
signal.alarm(120)  # 2 min timeout

try:
    start = time.time()
    data_iter = iter(data_loader)
    signal.alarm(0)
    
    iter_time = time.time() - start
    log(f"✓ STEP 9 - Iterator created in {iter_time:.1f}s")
    
    # Try loading a batch
    log("STEP 10: Load first batch")
    batch_start = time.time()
    batch = next(data_iter)
    batch_time = time.time() - batch_start
    log(f"✓ STEP 10 - Batch loaded in {batch_time:.1f}s")
    
    log("\n" + "="*60)
    log("ALL STEPS COMPLETED - NO DEADLOCK!")
    log("="*60)
    
except Exception as e:
    signal.alarm(0)
    log(f"\n✗ DEADLOCK at current step!")
    log(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if rank == 0 and 'wandb' in dir():
    wandb.finish()
