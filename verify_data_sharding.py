"""
Verify that data is properly sharded across ranks in distributed training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader

def test_sharding():
    """Test that different ranks see different data."""

    args = EEGDataArgs(
        data_path="/mnt/zehao/eeg_infra/data",
        batch_size=150,
        world_size=4,
    )

    # Simulate 4 ranks
    all_batches = {}
    for rank in range(4):
        args.rank = rank
        dataloader, _ = build_eeg_dataloader(args, train=True)

        # Get first batch from each rank
        first_batch = next(iter(dataloader))
        eeg, pos, mask, unmask = first_batch

        print(f"Rank {rank}:")
        print(f"  Total batches: {len(dataloader)}")
        print(f"  EEG shape: {eeg.shape}")
        print(f"  First sample hash: {hash(eeg[0].sum().item())}")
        print()

        all_batches[rank] = eeg[0].sum().item()

    # Check if all ranks have different data
    unique_hashes = len(set(all_batches.values()))
    print(f"\n{'='*50}")
    print(f"Unique first batches across {len(all_batches)} ranks: {unique_hashes}")

    if unique_hashes == len(all_batches):
        print("✅ SUCCESS: Each rank has different data (properly sharded)")
    else:
        print("❌ FAILURE: Some ranks have the same data (NOT properly sharded)")
        print(f"   Data hashes: {all_batches}")

    return unique_hashes == len(all_batches)

if __name__ == "__main__":
    success = test_sharding()
    sys.exit(0 if success else 1)
