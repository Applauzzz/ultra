"""
Benchmark EEG dataloader performance with config_train.yaml settings.
Measures time per batch loading step.
"""

import sys
import time
from pathlib import Path
import os

# Add ultra to path
sys.path.insert(0, str(Path(__file__).parent))

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader
import torch


def benchmark_dataloader(num_batches=50, warmup_batches=5):
    """
    Benchmark dataloader performance.

    Args:
        num_batches: Number of batches to measure (default 50)
        warmup_batches: Number of warmup batches (default 5)
    """
    print("=" * 70)
    print("EEG DataLoader Performance Benchmark")
    print("=" * 70)

    # Get CPU count
    cpu_count = os.cpu_count()
    num_workers = min(16, cpu_count)

    # Configuration matching config_train.yaml
    args = EEGDataArgs(
        data_path="/mnt/zehao/eeg_infra/data",
        subset="all",

        # Window and preprocessing (from config_train.yaml preprocessing section)
        window_duration=10000,
        clip=10.0,

        # Masking config (from config_train.yaml preprocessing.masking)
        masking_ratio=0.75,
        masking_window=200,
        masking_overlap=20,
        use_block_masking=False,  # block masking is more complex, start with simple

        # DataLoader config (from config_train.yaml data.loader)
        batch_size=300,  # From trainer.batch_size in config
        num_workers=num_workers,  # min(16, cpu_count)
        prefetch_factor=3,
        persistent_workers=True,

        # Distributed training
        world_size=1,
        rank=0,
    )

    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Subset: {args.subset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers} (CPU count: {cpu_count})")
    print(f"  Prefetch factor: {args.prefetch_factor}")
    print(f"  Persistent workers: {args.persistent_workers}")
    print(f"  Window duration: {args.window_duration}")
    print(f"  Masking ratio: {args.masking_ratio}")

    # Build dataloader
    print("\n" + "-" * 70)
    print("Building dataloader...")
    build_start = time.time()
    dataloader, state = build_eeg_dataloader(args, train=True)
    build_time = time.time() - build_start
    print(f"✓ Dataloader built in {build_time:.2f}s")
    print(f"  Total batches: {len(dataloader)}")

    # Warmup phase
    print("\n" + "-" * 70)
    print(f"Warmup phase ({warmup_batches} batches)...")
    warmup_start = time.time()

    for i, batch_data in enumerate(dataloader):
        if i >= warmup_batches:
            break
        eeg, pos, mask, unmask = batch_data
        # Just access the data to ensure it's loaded
        _ = eeg.shape

    warmup_time = time.time() - warmup_start
    print(f"✓ Warmup completed in {warmup_time:.2f}s")
    print(f"  Avg warmup time/batch: {warmup_time/warmup_batches:.4f}s")

    # Benchmark phase
    print("\n" + "-" * 70)
    print(f"Benchmark phase ({num_batches} batches)...")

    batch_times = []
    total_samples = 0
    total_time_points = 0

    overall_start = time.time()

    for i, batch_data in enumerate(dataloader):
        if i >= warmup_batches + num_batches:
            break
        if i < warmup_batches:
            continue

        batch_start = time.time()
        eeg, pos, mask, unmask = batch_data

        # Access data to ensure loading is complete
        batch_size, n_channels, time_points = eeg.shape
        _ = pos.shape

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        total_samples += batch_size
        total_time_points += batch_size * time_points

        if (i - warmup_batches) % 10 == 0:
            print(f"  Batch {i - warmup_batches}/{num_batches}: "
                  f"{batch_time:.4f}s, "
                  f"shape={eeg.shape}")

    overall_time = time.time() - overall_start

    # Statistics
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)

    import numpy as np
    batch_times = np.array(batch_times)

    print(f"\nTiming Statistics (n={len(batch_times)} batches):")
    print(f"  Mean time/batch:   {batch_times.mean():.4f}s")
    print(f"  Median time/batch: {np.median(batch_times):.4f}s")
    print(f"  Min time/batch:    {batch_times.min():.4f}s")
    print(f"  Max time/batch:    {batch_times.max():.4f}s")
    print(f"  Std dev:           {batch_times.std():.4f}s")

    print(f"\nPercentiles:")
    print(f"  P50: {np.percentile(batch_times, 50):.4f}s")
    print(f"  P90: {np.percentile(batch_times, 90):.4f}s")
    print(f"  P95: {np.percentile(batch_times, 95):.4f}s")
    print(f"  P99: {np.percentile(batch_times, 99):.4f}s")

    print(f"\nThroughput:")
    samples_per_sec = total_samples / overall_time
    print(f"  Samples/sec:       {samples_per_sec:.2f}")
    print(f"  Batches/sec:       {len(batch_times) / overall_time:.2f}")

    # Data volume
    avg_batch_size = total_samples / len(batch_times)
    avg_time_points_per_sample = total_time_points / total_samples
    data_per_batch_mb = (avg_batch_size * n_channels * avg_time_points_per_sample * 4) / (1024**2)
    throughput_mbps = data_per_batch_mb / batch_times.mean()

    print(f"\nData Volume:")
    print(f"  Avg samples/batch:     {avg_batch_size:.0f}")
    print(f"  Channels:              {n_channels}")
    print(f"  Time points/sample:    {avg_time_points_per_sample:.0f}")
    print(f"  Data per batch:        {data_per_batch_mb:.2f} MB")
    print(f"  Throughput:            {throughput_mbps:.2f} MB/s")

    print("\n" + "=" * 70)
    print(f"✅ Benchmark Complete!")
    print("=" * 70)

    # Summary for quick reference
    print(f"\n📊 Quick Summary:")
    print(f"   ⏱️  Time per batch:  {batch_times.mean():.4f}s ± {batch_times.std():.4f}s")
    print(f"   🚀 Throughput:       {samples_per_sec:.1f} samples/sec")
    print(f"   💾 Data rate:        {throughput_mbps:.1f} MB/s")

    return {
        'mean_time': batch_times.mean(),
        'std_time': batch_times.std(),
        'median_time': np.median(batch_times),
        'throughput_samples_per_sec': samples_per_sec,
        'throughput_mbps': throughput_mbps,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark EEG DataLoader')
    parser.add_argument('--num-batches', type=int, default=50,
                        help='Number of batches to benchmark (default: 50)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup batches (default: 5)')

    args = parser.parse_args()

    try:
        benchmark_dataloader(num_batches=args.num_batches, warmup_batches=args.warmup)
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Benchmark FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
