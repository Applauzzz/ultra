#!/usr/bin/env python3
"""Debug training - print and flush at every step to locate hang."""

import os
import sys
import time
import torch
import torch.distributed as dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def log(msg, rank=None):
    """Log with immediate flush."""
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    log(f"Rank {rank}/{world_size}, Local rank {local_rank}", rank)
    
    if world_size > 1:
        log("Initializing process group...", rank)
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        log("Process group initialized", rank)
    
    return rank, local_rank, world_size

def main():
    log("=" * 60)
    log("DEBUG TRAINING - STEP BY STEP")
    log("=" * 60)
    
    # Step 1: Distributed init
    log("STEP 1: Initializing distributed...")
    rank, local_rank, world_size = init_distributed()
    log("✓ STEP 1 DONE")
    
    # Step 2: Import modules
    log("STEP 2: Importing modules...")
    from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader
    from ultra.model_eeg import MAEModelArgs, MAE
    log("✓ STEP 2 DONE")
    
    # Step 3: Build DataLoader
    log("STEP 3: Building DataLoader...")
    data_args = EEGDataArgs(
        data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
        subset="all",
        batch_size=4,  # Very small batch
        num_workers=0,  # No workers to avoid deadlock
        persistent_workers=False,
        world_size=world_size,
        rank=rank,
    )
    log("  Building dataloader...")
    data_loader, _ = build_eeg_dataloader(data_args, train=True)
    log(f"✓ STEP 3 DONE - {len(data_loader)} batches")
    
    # Step 4: Build model
    log("STEP 4: Building model...")
    model_args = MAEModelArgs()
    model = MAE(model_args)
    log(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    model = model.cuda()
    log("✓ STEP 4 DONE")
    
    # Step 5: Build optimizer
    log("STEP 5: Building optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    log("✓ STEP 5 DONE")
    
    # Step 6: Create iterator
    log("STEP 6: Creating DataLoader iterator...")
    log("  Calling iter(data_loader)...")
    data_iter = iter(data_loader)
    log("✓ STEP 6 DONE")
    
    # Step 7: Load first batch
    log("STEP 7: Loading first batch...")
    log("  Calling next(data_iter)...")
    eeg, pos, batch_mask, batch_unmask = next(data_iter)
    log(f"  Batch loaded: {eeg.shape}")
    log("  Moving to GPU...")
    eeg = eeg.cuda()
    pos = pos.cuda()
    batch_mask = batch_mask.cuda()
    batch_unmask = batch_unmask.cuda()
    log("✓ STEP 7 DONE")
    
    # Step 8: Forward pass
    log("STEP 8: Running forward pass...")
    model.train()
    log("  Calling model.forward()...")
    loss = model(eeg, pos, batch_mask, batch_unmask)
    log(f"✓ STEP 8 DONE - Loss: {loss.item():.4f}")
    
    # Step 9: Backward
    log("STEP 9: Running backward...")
    optimizer.zero_grad()
    log("  Calling loss.backward()...")
    loss.backward()
    log("✓ STEP 9 DONE")
    
    # Step 10: Optimizer step
    log("STEP 10: Running optimizer step...")
    log("  Calling optimizer.step()...")
    optimizer.step()
    log("✓ STEP 10 DONE")
    
    # Step 11: Second iteration
    log("STEP 11: Testing second iteration...")
    log("  Loading batch 2...")
    eeg, pos, batch_mask, batch_unmask = next(data_iter)
    eeg = eeg.cuda()
    pos = pos.cuda()
    batch_mask = batch_mask.cuda()
    batch_unmask = batch_unmask.cuda()
    log("  Forward pass...")
    loss = model(eeg, pos, batch_mask, batch_unmask)
    log(f"✓ STEP 11 DONE - Loss: {loss.item():.4f}")
    
    log("=" * 60)
    log("ALL STEPS COMPLETED SUCCESSFULLY!")
    log("Training pipeline is working correctly.")
    log("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        log(f"ERROR at rank {rank}: {e}", rank)
        import traceback
        traceback.print_exc()
        sys.exit(1)
