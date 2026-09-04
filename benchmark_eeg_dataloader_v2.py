"""
Benchmark EEG dataloader performance - measuring actual loading time.
Uses a fresh iterator for each batch to avoid prefetching effects.
"""

import sys
import time
from pathlib import Path
import os

# Add ultra to path
sys.path.insert(0, str(Path(__file__).parent))

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader
import torch


def benchmark_dataloader_realistic(num_batches=50):
    """
    Benchmark dataloader with realistic timing (includes prefetch behavior).
    """
    print("=" * 70)
    print("EEG DataLoader Realistic Performance Benchmark")
    print("=" * 70)

    # Get CPU count
    cpu_count = os.cpu_count()
    num_workers = min(16, cpu_count)

    # Configuration matching config_train.yaml
    args = EEGDataArgs(
        data_path="/mnt/zehao/eeg_infra/data",
        subset="all",
        window_duration=10000,
        clip=10.0,
        masking_ratio=0.75,
        masking_window=200,
        masking_overlap=20,
        use_block_masking=False,
        batch_size=300,
        num_workers=num_workers,
        prefetch_factor=3,
        persistent_workers=True,
        world_size=1,
        rank=0,
    )

    print(f"\nConfiguration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Prefetch factor: {args.prefetch_factor}")
    print(f"  Persistent workers: {args.persistent_workers}")

    # Build dataloader
    print("\n" + "-" * 70)
    print("Building dataloader...")
    dataloader, state = build_eeg_dataloader(args, train=True)
    print(f"✓ Total batches available: {len(dataloader)}")

    # Test different scenarios
    print("\n" + "=" * 70)
    print("Scenario 1: Continuous iteration (with prefetching)")
    print("=" * 70)

    iter_start = time.time()
    batch_times = []
    total_samples = 0

    data_iter = iter(dataloader)

    for i in range(min(num_batches, len(dataloader))):
        batch_start = time.time()

        try:
            eeg, pos, mask, unmask = next(data_iter)
        except StopIteration:
            break

        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)

        batch_size = eeg.shape[0]
        total_samples += batch_size

        if i % 10 == 0:
            print(f"  Batch {i}: {batch_time*1000:.2f}ms, shape={eeg.shape}")

    iter_total = time.time() - iter_start

    import numpy as np
    batch_times = np.array(batch_times)

    print(f"\n📊 Results (Continuous Iteration):")
    print(f"  Total time for {len(batch_times)} batches: {iter_total:.4f}s")
    print(f"  Time per batch (mean): {batch_times.mean()*1000:.2f}ms")
    print(f"  Time per batch (median): {np.median(batch_times)*1000:.2f}ms")
    print(f"  Time per batch (p95): {np.percentile(batch_times, 95)*1000:.2f}ms")
    print(f"  Throughput: {total_samples/iter_total:.1f} samples/sec")
    print(f"  Batches/sec: {len(batch_times)/iter_total:.2f}")

    # Test scenario 2: First batch timing (cold start)
    print("\n" + "=" * 70)
    print("Scenario 2: First batch timing (cold start, no prefetch)")
    print("=" * 70)

    # Create fresh dataloader
    dataloader2, _ = build_eeg_dataloader(args, train=True)

    first_batch_times = []
    for trial in range(5):
        # Fresh iterator for each trial
        data_iter = iter(dataloader2)

        start = time.time()
        eeg, pos, mask, unmask = next(data_iter)
        first_time = time.time() - start

        first_batch_times.append(first_time)
        print(f"  Trial {trial + 1}: {first_time*1000:.2f}ms, shape={eeg.shape}")

        # Let the iterator be garbage collected
        del data_iter

    first_batch_times = np.array(first_batch_times)
    print(f"\n📊 First Batch Statistics:")
    print(f"  Mean: {first_batch_times.mean()*1000:.2f}ms")
    print(f"  Median: {np.median(first_batch_times)*1000:.2f}ms")
    print(f"  Min: {first_batch_times.min()*1000:.2f}ms")
    print(f"  Max: {first_batch_times.max()*1000:.2f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n✅ DataLoader Performance:")
    print(f"   Steady-state (with prefetch): {batch_times.mean()*1000:.2f}ms/batch")
    print(f"   Cold start (first batch):     {first_batch_times.mean()*1000:.2f}ms/batch")
    print(f"   Throughput:                   {total_samples/iter_total:.1f} samples/sec")

    # Estimate training step time
    print(f"\n💡 Training Step Estimate:")
    print(f"   Data loading time:  {batch_times.mean()*1000:.2f}ms")
    print(f"   (GPU forward+backward will add more time)")

    return {
        'steady_state_ms': batch_times.mean() * 1000,
        'cold_start_ms': first_batch_times.mean() * 1000,
        'throughput_samples_per_sec': total_samples / iter_total,
    }


if __name__ == "__main__":
    try:
        benchmark_dataloader_realistic(num_batches=50)
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Benchmark FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
