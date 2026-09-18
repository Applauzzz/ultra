#!/usr/bin/env python3
"""Minimal training test - find where it hangs."""

import os
import sys
import time
import torch
import torch.distributed as dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader
from ultra.model_eeg import MAEModelArgs, MAE

def checkpoint(msg):
    """Print checkpoint with timestamp."""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] ✓ {msg}")
        sys.stdout.flush()

def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def test_minimal_train():
    rank, local_rank, world_size = init_distributed()
    
    checkpoint("Distributed initialized")
    
    # Build dataloader
    data_args = EEGDataArgs(
        data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
        subset="all",
        batch_size=16,
        num_workers=2,
        persistent_workers=False,
        world_size=world_size,
        rank=rank,
    )
    
    checkpoint("Building DataLoader...")
    data_loader, _ = build_eeg_dataloader(data_args, train=True)
    checkpoint(f"DataLoader built: {len(data_loader)} batches")
    
    # Build model (without FSDP first)
    checkpoint("Building model...")
    model_args = MAEModelArgs()
    model = MAE(model_args).cuda()
    checkpoint(f"Model built: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    # Build optimizer
    checkpoint("Building optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint("Optimizer built")
    
    # Test first batch
    checkpoint("Creating DataLoader iterator...")
    data_iter = iter(data_loader)
    checkpoint("Iterator created")
    
    checkpoint("Loading first batch...")
    eeg, pos, batch_mask, batch_unmask = next(data_iter)
    checkpoint(f"First batch loaded: {eeg.shape}")
    
    eeg = eeg.cuda()
    pos = pos.cuda()
    batch_mask = batch_mask.cuda()
    batch_unmask = batch_unmask.cuda()
    checkpoint("Batch moved to GPU")
    
    # Forward pass
    checkpoint("Running forward pass...")
    model.train()
    loss = model(eeg, pos, batch_mask, batch_unmask)
    checkpoint(f"Forward pass done: loss={loss.item():.4f}")
    
    # Backward pass
    checkpoint("Running backward pass...")
    optimizer.zero_grad()
    loss.backward()
    checkpoint("Backward pass done")
    
    # Optimizer step
    checkpoint("Running optimizer step...")
    optimizer.step()
    checkpoint("Optimizer step done")
    
    checkpoint("=" * 50)
    checkpoint("MINIMAL TRAINING TEST PASSED!")
    checkpoint("=" * 50)

if __name__ == "__main__":
    try:
        test_minimal_train()
    except Exception as e:
        rank = int(os.environ.get("RANK", 0))
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
