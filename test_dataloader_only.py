#!/usr/bin/env python3
"""Test EEG DataLoader independently without training."""

import time
import sys
from pathlib import Path

# Add ultra to path
sys.path.insert(0, str(Path(__file__).parent))

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

def test_dataloader():
    print("=" * 60)
    print("EEG DataLoader Standalone Test")
    print("=" * 60)
    
    # Configure data loading
    args = EEGDataArgs(
        data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
        subset="all",  # Test with 'all' dataset
        batch_size=16,  # Small batch for testing
        num_workers=0,  # Start with 0 workers (main process only)
        prefetch_factor=2,
        persistent_workers=False,
        world_size=1,
        rank=0,
    )
    
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Subset: {args.subset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Persistent workers: {args.persistent_workers}")
    
    # Build dataloader
    print(f"\n{'Building DataLoader':-^60}")
    start = time.time()
    try:
        data_loader, state = build_eeg_dataloader(args, train=True)
        build_time = time.time() - start
        print(f"✓ DataLoader built in {build_time:.2f}s")
        print(f"  Total batches: {len(data_loader)}")
    except Exception as e:
        print(f"✗ Failed to build DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test iteration
    print(f"\n{'Testing iteration (first 5 batches)':-^60}")
    try:
        iter_start = time.time()
        data_iter = iter(data_loader)
        print(f"✓ DataLoader iterator created in {time.time() - iter_start:.2f}s")
        
        for i in range(min(5, len(data_loader))):
            batch_start = time.time()
            eeg, pos, batch_mask, batch_unmask = next(data_iter)
            batch_time = time.time() - batch_start
            
            print(f"  Batch {i+1}:")
            print(f"    EEG shape: {eeg.shape}")
            print(f"    Positions shape: {pos.shape}")
            print(f"    Mask shape: {batch_mask.shape}")
            print(f"    Unmask shape: {batch_unmask.shape}")
            print(f"    Load time: {batch_time:.3f}s")
            
            if i == 0:
                print(f"    EEG dtype: {eeg.dtype}, range: [{eeg.min():.2f}, {eeg.max():.2f}]")
        
        print(f"\n✓ Successfully loaded {min(5, len(data_loader))} batches")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed during iteration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_workers():
    """Test with multiple workers."""
    print("\n" + "=" * 60)
    print("Testing with workers")
    print("=" * 60)
    
    for num_workers in [1, 2, 4]:
        print(f"\n{'Testing with ' + str(num_workers) + ' workers':-^60}")
        
        args = EEGDataArgs(
            data_path="/mnt_upfs/zehao/database/egg/reve_official_data",
            subset="all",
            batch_size=16,
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=False,
            world_size=1,
            rank=0,
        )
        
        try:
            start = time.time()
            data_loader, _ = build_eeg_dataloader(args, train=True)
            build_time = time.time() - start
            print(f"✓ Built in {build_time:.2f}s")
            
            # Test first batch
            iter_start = time.time()
            data_iter = iter(data_loader)
            iter_time = time.time() - iter_start
            print(f"✓ Iterator created in {iter_time:.2f}s")
            
            batch_start = time.time()
            eeg, pos, batch_mask, batch_unmask = next(data_iter)
            batch_time = time.time() - batch_start
            print(f"✓ First batch loaded in {batch_time:.2f}s, shape: {eeg.shape}")
            
        except Exception as e:
            print(f"✗ Failed with {num_workers} workers: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    # Test basic dataloader (no workers)
    success = test_dataloader()
    
    if success:
        print("\n" + "=" * 60)
        print("Basic test PASSED - proceeding to worker tests")
        print("=" * 60)
        test_with_workers()
    else:
        print("\n" + "=" * 60)
        print("Basic test FAILED - skipping worker tests")
        print("=" * 60)
        sys.exit(1)
