#!/usr/bin/env python3
"""Test EEG DataLoader in distributed environment."""

import os
import sys
import time
import torch
import torch.distributed as dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

def init_distributed():
    """Initialize distributed training."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def test_distributed_dataloader():
    rank, local_rank, world_size = init_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("Distributed DataLoader Test")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"GPUs: {torch.cuda.device_count()}")
    
    # Configure data loading
    args = EEGDataArgs(
        data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
        subset="all",
        batch_size=16,
        num_workers=2,  # Start with 2 workers per GPU
        prefetch_factor=2,
        persistent_workers=False,
        world_size=world_size,
        rank=rank,
    )
    
    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Num workers: {args.num_workers}")
        print(f"  Total workers: {args.num_workers * world_size}")
    
    # Build dataloader
    if rank == 0:
        print(f"\n{'Building DataLoader':-^60}")
    
    start = time.time()
    try:
        data_loader, state = build_eeg_dataloader(args, train=True)
        build_time = time.time() - start
        
        if rank == 0:
            print(f"✓ DataLoader built in {build_time:.2f}s")
            print(f"  Batches per GPU: {len(data_loader)}")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Failed to build DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Synchronize before iteration
    if world_size > 1:
        dist.barrier()
    
    # Test iteration
    if rank == 0:
        print(f"\n{'Testing iteration (first 3 batches per GPU)':-^60}")
    
    try:
        iter_start = time.time()
        data_iter = iter(data_loader)
        iter_time = time.time() - iter_start
        
        print(f"[Rank {rank}] ✓ Iterator created in {iter_time:.2f}s")
        
        for i in range(min(3, len(data_loader))):
            batch_start = time.time()
            eeg, pos, batch_mask, batch_unmask = next(data_iter)
            batch_time = time.time() - batch_start
            
            print(f"[Rank {rank}] Batch {i+1}: shape={eeg.shape}, time={batch_time:.3f}s")
        
        if rank == 0:
            print(f"\n✓ All ranks successfully loaded batches")
        
        return True
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Failed during iteration: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if world_size > 1:
            dist.barrier()

if __name__ == "__main__":
    success = test_distributed_dataloader()
    
    if not success:
        sys.exit(1)
