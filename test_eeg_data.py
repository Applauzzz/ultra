"""
Quick test script for EEG data loading in ultra framework.
"""

import sys
from pathlib import Path

# Add ultra to path
sys.path.insert(0, str(Path(__file__).parent))

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader


def test_eeg_dataloader():
    """Test EEG dataloader with demo data."""
    print("=" * 60)
    print("Testing EEG Data Loading")
    print("=" * 60)

    # Configuration
    args = EEGDataArgs(
        data_path="/mnt/zehao/eeg_infra/data",
        subset="all",
        batch_size=448,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        world_size=1,
        rank=0,
    )

    print(f"\nData path: {args.data_path}")
    print(f"Subset: {args.subset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")

    # Build dataloader
    print("\n" + "-" * 60)
    print("Building dataloader...")
    dataloader, state = build_eeg_dataloader(args, train=True)
    print(f"✓ Dataloader built successfully!")
    print(f"  Total batches: {len(dataloader)}")

    # Test loading
    print("\n" + "-" * 60)
    print("Testing batch loading...")

    for batch_idx, batch_data in enumerate(dataloader):
        eeg, pos, mask, unmask = batch_data

        print(f"\nBatch {batch_idx}:")
        print(f"  EEG shape: {eeg.shape}")
        print(f"    - Expected: (batch_size, channels, time)")
        print(f"    - Dtype: {eeg.dtype}")
        print(f"    - Device: {eeg.device}")
        print(f"    - Value range: [{eeg.min():.2f}, {eeg.max():.2f}]")

        print(f"  Positions shape: {pos.shape}")
        print(f"    - Expected: (batch_size, channels, 3)")
        print(f"    - Dtype: {pos.dtype}")

        print(f"  Mask indices shape: {mask.shape}")
        print(f"  Unmask indices shape: {unmask.shape}")

        # Sanity checks
        assert eeg.ndim == 3, f"EEG should be 3D, got {eeg.ndim}D"
        assert pos.ndim == 3, f"Positions should be 3D, got {pos.ndim}D"
        assert pos.shape[-1] == 3, f"Positions last dim should be 3 (x,y,z), got {pos.shape[-1]}"

        if batch_idx >= 2:
            print(f"\n... (showing first 3 batches only)")
            break

    print("\n" + "=" * 60)
    print("✅ EEG data loading test PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_eeg_dataloader()
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Test FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
