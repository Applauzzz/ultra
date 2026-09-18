"""
EEG data loading for ultra framework.
Adapted from eeg_infra/src/utils/data_loading.py to work with ultra's distributed training.
"""

import ast
import csv
import gc
import random
from collections import defaultdict
from dataclasses import dataclass, field
import os
from os.path import join as pjoin
from typing import Dict, Any, Optional, Iterator, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.utils.data import DataLoader, Dataset, Sampler
import logging

logger = logging.getLogger(__name__)


# ==================== Data Args ====================
@dataclass
class EEGDataArgs:
    """Configuration for EEG data loading."""

    # Data paths
    data_path: str = "/path/to/eeg_data"  # Root path containing recordings/, positions/, stats/, csv_recordings/

    # Data subset selection
    subset: str = "all"  # "all", "small", or "open"

    # Window and preprocessing
    window_duration: int = 10000  # 10 seconds at 1000Hz
    clip: float = 10.0  # Clip normalized values to [-clip, clip]

    # Masking config
    masking_ratio: float = 0.75
    masking_window: int = 200  # Patch size
    masking_overlap: int = 20   # Patch overlap
    use_block_masking: bool = False
    radius_spat_mask: float = 0.1
    radius_temp_mask: int = 5
    dropout_ratio: float = 0.0
    dropout_radius: float = 0.05

    # DataLoader config
    batch_size: int = 448
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True  # Recommended for EEG

    # Distributed training
    world_size: int = 1
    rank: int = 0


# ==================== State Management ====================
@dataclass
class EEGDataLoaderState:
    """State for resuming EEG data loading."""

    epoch: int = 0
    batch_idx: int = 0
    sampler_seed: int = 42

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "sampler_seed": self.sampler_seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.epoch = state_dict["epoch"]
        self.batch_idx = state_dict["batch_idx"]
        self.sampler_seed = state_dict["sampler_seed"]


# ==================== Helper Functions ====================

# Subset definitions (from original eeg_infra)
SUBSET_VAL = [78, 83, 84, 93, 97]
SUBSET_TRAIN = [72, 73, 74, 75, 77, 79, 80, 81, 82, 85, 86, 87, 88, 90, 92, 98, 99, 100, 101, 102, 103, 104]

OPEN_VAL = [227, 113, 137, 178, 110, 90, 358, 160, 156, 233, 131, 81, 394, 229, 312, 401, 437, 267, 116, 132]  # Truncated for brevity
OPEN_TRAIN = [196, 285, 155, 219, 127, 228, 121, 327, 268, 82, 145, 105, 194, 435, 192, 262, 436, 199, 314]  # Truncated


def _read_csv(path: str) -> list:
    """Read CSV file and return rows as list of dicts."""
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f, delimiter=","))


def _filter_by_recording_set(rows: list, recording_set: set) -> list:
    """Filter rows by recording index."""
    allowed = set(recording_set)
    return [r for r in rows if int(r["big_recording_index"]) in allowed]


def _group_by(rows: list, key: str) -> dict:
    """Group rows by a key."""
    groups = defaultdict(list)
    for row in rows:
        groups[int(row[key])].append(row)
    return dict(groups)


def _make_window_keys(big_rec_index, session_idx, start, end, window_duration, chans_to_remove=None):
    """Create window segment keys."""
    offsets = np.arange(start, end - window_duration, window_duration)
    if chans_to_remove is None:
        return [f"{big_rec_index}_-_{session_idx}_-_{w}" for w in offsets]
    return [f"{big_rec_index}_-_{session_idx}_-_{w}_-_{chans_to_remove}" for w in offsets]


def compute_group_segments(data, data_big, window_duration):
    """Compute channel-grouped window segment keys."""
    chan_aggregates = _group_by(data_big, "n_chans")
    dict_groups = defaultdict(list)
    dict_groups_incorrect = defaultdict(list)

    for n_chans, recordings in chan_aggregates.items():
        windows = []
        for recording_data in recordings:
            b_idx = recording_data["big_recording_index"]
            index_data = [d for d in data if d["big_recording_index"] == b_idx]

            # Flag faulty reduce for removal
            if index_data[-1]["flag_reduce"] != index_data[-2]["flag_reduce"]:
                index_data[-1]["flag_remove"] = "True"

            # Compute cumulative start/end
            corrected_starts = [0] + np.cumsum([int(d["duration"]) for d in index_data[:-1]]).tolist()
            for d, s in zip(index_data, corrected_starts):
                d["start"] = s
                d["end"] = s + int(d["duration"])

            # Filter sessions shorter than one window
            index_data = [d for d in index_data if int(d["duration"]) > window_duration]

            # Create window keys
            for i, d in enumerate(index_data):
                if d["flag_remove"] == "True":
                    continue

                n_to_remove = int(d["n_chans_to_remove"])
                if n_to_remove == 0:
                    windows += _make_window_keys(b_idx, i, d["start"], d["end"], window_duration)
                else:
                    chans_str = "/".join(str(x) for x in ast.literal_eval(str(d["flag_reduce"])))
                    keys = _make_window_keys(b_idx, i, d["start"], d["end"], window_duration, chans_str)
                    dict_groups_incorrect[n_chans - n_to_remove] += keys

        dict_groups[n_chans] += windows

    # Merge faulty-channel windows
    for effective_chans, keys in dict_groups_incorrect.items():
        dict_groups[effective_chans] += keys

    segments = [key for keys in dict_groups.values() for key in keys]
    dict_groups = dict(sorted(dict_groups.items()))

    return dict_groups, segments


def spatial_masking(C, masking_ratio, radius, precomp_masked_indices=None):
    """Create spatial masks using KD-tree."""
    n_channels = C.shape[0]
    n_masked_channels = int(masking_ratio * n_channels)
    mask = np.zeros(n_channels, dtype=bool)

    if precomp_masked_indices is not None:
        mask[precomp_masked_indices] = True

    kdtree = KDTree(C)
    unmasked_indices = np.where(~mask)[0]

    while len(unmasked_indices) > 0 and np.sum(mask) < n_masked_channels:
        unmasked_indices = np.where(~mask)[0]
        _index = np.random.choice(unmasked_indices)
        _channel = C[_index]
        _neighbors = kdtree.query_ball_point(_channel, radius)
        mask[_neighbors] = True

    return np.where(mask)[0]


def create_block_masks(n_chans, masking_ratio, radius_spat_mask, radius_temp_mask,
                       num_patches, pos, dropout_ratio, dropout_ratio_radius):
    """Create block masks for training."""
    num_block_patches = num_patches // radius_temp_mask
    num_isolated_patches = num_patches % radius_temp_mask
    num_masked_chans = int(masking_ratio * n_chans)
    chan_indices = np.arange(n_chans)

    # Dropout masks
    if dropout_ratio > 0:
        dropout_channels = spatial_masking(pos, dropout_ratio, dropout_ratio_radius)[:int(dropout_ratio * n_chans)]
    else:
        dropout_channels = None

    # Block masks
    block_masks = [
        spatial_masking(pos, masking_ratio, radius_spat_mask, dropout_channels)[:num_masked_chans]
        for _ in range(num_block_patches)
    ]
    idx_block = np.repeat(np.array(block_masks)[:, np.newaxis, :], radius_temp_mask, axis=1).reshape(-1, num_masked_chans)

    # Isolated masks
    isolated_masks = [
        spatial_masking(pos, masking_ratio, radius_spat_mask)[:num_masked_chans]
        for _ in range(num_isolated_patches)
    ]
    idx_isolated = np.array(isolated_masks)

    mask = np.concatenate([idx_block, idx_isolated], axis=0)
    unmask = np.array([np.setdiff1d(chan_indices, masked_indices, assume_unique=True) for masked_indices in mask])

    flat_indices = np.arange(mask.shape[0])[:, np.newaxis]
    masked_indices = (n_chans * flat_indices + mask).ravel()
    unmasked_indices = (n_chans * flat_indices + unmask).ravel()

    np.random.shuffle(masked_indices)
    np.random.shuffle(unmasked_indices)

    return torch.from_numpy(masked_indices), torch.from_numpy(unmasked_indices)


# ==================== Dataset ====================

class EEGDataset(Dataset):
    """EEG Dataset with memmap loading and masking."""

    def __init__(self, segments, groups, data_big, data_stats, recordings_path,
                 window_duration, clip, block_masking, masking_window, masking_overlap,
                 masking_ratio, radius_spat_mask, radius_temp_mask, dropout_ratio,
                 dropout_radius, no_masking=False, manual_seed=False):

        self.path = recordings_path
        self.path_pos = "/".join(self.path.split("/")[:-1] + ["positions"])
        self.path_stats = "/".join(self.path.split("/")[:-1] + ["stats"])

        self.segments = segments
        self.groups = groups
        self.window_duration = window_duration
        self.clip = clip
        self.block_masking = block_masking
        self.masking_window = masking_window
        self.masking_overlap = masking_overlap
        self.masking_ratio = masking_ratio
        self.no_masking = no_masking

        if self.block_masking:
            self.radius_spat_mask = radius_spat_mask
            self.radius_temp_mask = radius_temp_mask
            self.dropout_ratio = dropout_ratio
            self.dropout_radius = dropout_radius
        else:
            self.manual_seed = manual_seed

        self.data_big = data_big
        self.data_stats = data_stats
        self.init_files_pos()

        self.counter = 0
        self.max_counter = 500000

    def init_files_pos(self):
        """Initialize memmap files and position file paths (lazy loading)."""
        import os
        self.files = {}
        self.positions_cache = {}  # Will store loaded positions on-demand
        self.position_paths = {}  # Store file paths for lazy loading

        for recording_data in self.data_big:
            r_i = int(recording_data["big_recording_index"])
            r_t = int(recording_data["duration"])
            r_c = int(recording_data["n_chans"])

            # Check if file exists before trying to open
            eeg_file = pjoin(self.path, f"recording_-_eeg_-_{r_i}.npy")
            pos_file = pjoin(self.path_pos, f"recording_-_positions_-_{r_i}.npy")

            if not os.path.exists(eeg_file):
                logger.warning(f"Skipping recording {r_i}: EEG file not found at {eeg_file}")
                continue
            if not os.path.exists(pos_file):
                logger.warning(f"Skipping recording {r_i}: Position file not found at {pos_file}")
                continue

            memmap_array = np.memmap(
                eeg_file,
                mode="r",
                shape=(r_t, r_c),
                dtype="float32",
            )
            self.files[r_i] = memmap_array
            # Store path instead of loading immediately (lazy loading)
            self.position_paths[r_i] = pos_file

        self.stats = {}
        for recording_data in self.data_stats:
            r_i = int(recording_data["big_recording_index"])

            # Skip if recording was not loaded
            if r_i not in self.files:
                continue

            n_s = int(recording_data["n_sessions"])
            n_c = int(recording_data["n_chans"])

            stats_file = pjoin(self.path_stats, f"recording_-_stats_-_{r_i}.npy")
            if not os.path.exists(stats_file):
                logger.warning(f"Skipping stats for recording {r_i}: file not found at {stats_file}")
                continue

            memmap_stats = np.memmap(
                stats_file,
                mode="r",
                shape=(n_s, 2, n_c),
                dtype="float32",
            )
            self.stats[r_i] = memmap_stats

    def del_files(self):
        """Close and delete memmap files."""
        for k, v in self.files.items():
            v._mmap.close()
            del v
        for k, v in self.stats.items():
            v._mmap.close()
            del v
        del self.files, self.positions_cache, self.stats
        gc.collect()

    def __getitem__(self, index):
        """Get a single EEG window with masking."""
        if self.counter >= self.max_counter:
            self.del_files()
            self.init_files_pos()
            self.counter = 0
        self.counter += 1

        # Parse index
        index_ = index.split("_-_")
        if len(index_) == 3:
            b_rec, rec, offset = index_
            faulty_chans = False
        elif len(index_) == 4:
            b_rec, rec, offset, faulty_chans = index_
            faulty_chans = [int(x) for x in faulty_chans.split("/")]
        else:
            raise ValueError(f"Invalid index format: {index}")

        b_rec, rec, offset = int(b_rec), int(rec), int(offset)

        # Lazy load positions on first access
        if b_rec not in self.positions_cache:
            self.positions_cache[b_rec] = np.load(self.position_paths[b_rec])

        positions = self.positions_cache[b_rec].copy()
        eeg = self.files[b_rec][offset : offset + self.window_duration].copy()

        # Mask faulty channels
        if faulty_chans:
            mask = np.ones(positions.shape[0], dtype=bool)
            mask[faulty_chans] = False
            eeg = eeg[:, mask]
            positions = positions[mask]

        # Normalization
        if b_rec in self.stats:
            # Use pre-computed stats if available
            stats = self.stats[b_rec][rec]
            if faulty_chans:
                stats = stats[:, mask]
            eps = 1e-10 if any(stats[1] == 0) else 0
            eeg -= stats[0]
            eeg /= stats[1] + eps
        else:
            # Fallback: compute stats on-the-fly (per-channel z-normalization)
            # This is slower but works when stats files are missing
            mean = eeg.mean(axis=0, keepdims=True)
            std = eeg.std(axis=0, keepdims=True) + 1e-10
            eeg = (eeg - mean) / std

        eeg = torch.from_numpy(eeg).transpose(0, 1)  # (C, T)

        if self.no_masking:
            return eeg.float().clip(-self.clip, self.clip), torch.from_numpy(positions).float()

        # Apply masking
        if self.block_masking:
            patches = eeg.unfold(dimension=1, size=self.masking_window, step=self.masking_window - self.masking_overlap)
            c, h, p = patches.shape
            batch_mask, batch_unmask = create_block_masks(
                c, self.masking_ratio, self.radius_spat_mask, self.radius_temp_mask,
                h, positions, self.dropout_ratio, self.dropout_radius
            )
        else:
            patches = eeg.unfold(dimension=1, size=self.masking_window, step=self.masking_window - self.masking_overlap)
            c, h, p = patches.shape
            num_patches = c * h
            num_masks = int(self.masking_ratio * num_patches)
            if self.manual_seed is not None and self.manual_seed is not False:
                torch.manual_seed(int(self.manual_seed))
            rand_indices = torch.rand(1, num_patches).argsort(dim=-1)
            batch_mask = rand_indices[:, :num_masks].squeeze(0)
            batch_unmask = rand_indices[:, num_masks:].squeeze(0)

        return (
            eeg.float().clip(-self.clip, self.clip),
            torch.from_numpy(positions).float(),
            batch_mask,
            batch_unmask,
        )

    def __len__(self):
        return len(self.segments)


# ==================== Sampler ====================

def get_local_batch_size(c, global_batch_size):
    """Compute local batch size based on channel count."""
    C = 16 * global_batch_size
    return max(1, int(C // c))


class GroupedSampler(Sampler):
    """Sampler that groups EEG windows by channel count for efficient batching."""

    def __init__(self, dataset, batch_size, drop_last, n_gpu, rank=0, mode="train"):
        self.dataset = dataset
        self.segments = self.dataset.segments
        self.groups = self.dataset.groups
        self.mode = mode
        self.global_batch_size = batch_size
        self.drop_last = 0 if drop_last else 1
        self.n_gpu = n_gpu
        self.rank = rank

        # Recording slices for locality
        self.recording_slices = {
            n_chans: self._recording_slices(group_indices)
            for n_chans, group_indices in self.groups.items()
        }
        self.indices = []

    @staticmethod
    def _recording_index(segment):
        return segment.split("_-_", 1)[0]

    @classmethod
    def _recording_slices(cls, group_indices):
        slices = defaultdict(list)
        if not group_indices:
            return slices

        start = 0
        recording_index = cls._recording_index(group_indices[0])
        for index in range(1, len(group_indices)):
            next_recording_index = cls._recording_index(group_indices[index])
            if next_recording_index != recording_index:
                slices[recording_index].append((start, index))
                start = index
                recording_index = next_recording_index
        slices[recording_index].append((start, len(group_indices)))
        return slices

    def _recording_local_batches(self, n_chans):
        """Batch one channel group while keeping adjacent file offsets together."""
        group_indices = self.groups[n_chans]
        local_batch_size = get_local_batch_size(n_chans, self.global_batch_size)
        recording_slices = self.recording_slices[n_chans]
        recording_indices = list(recording_slices)
        random.shuffle(recording_indices)

        locality_runs = []
        pending = []
        for recording_index in recording_indices:
            recording_batches = []
            for start, end in recording_slices[recording_index]:
                cursor = start

                if pending:
                    take = min(local_batch_size - len(pending), end - cursor)
                    pending.extend(group_indices[cursor : cursor + take])
                    cursor += take
                    if len(pending) == local_batch_size:
                        recording_batches.append(pending)
                        pending = []

                while cursor + local_batch_size <= end:
                    recording_batches.append(group_indices[cursor : cursor + local_batch_size])
                    cursor += local_batch_size

                if cursor < end:
                    pending.extend(group_indices[cursor:end])

            if recording_batches:
                locality_runs.append(recording_batches)

        if pending and self.drop_last:
            locality_runs.append([pending])

        return locality_runs

    def __iter__(self):
        indices = []
        n_chans = list(self.groups.keys())

        if self.mode == "train":
            n_chans = [c for c in n_chans if c > 6]
            locality_runs = []
            for c in n_chans:
                locality_runs.extend(self._recording_local_batches(c))

            random.shuffle(locality_runs)
            indices = [batch for locality_run in locality_runs for batch in locality_run]
        else:
            for c in n_chans:
                group_indices = self.groups[c]
                indices += [
                    group_indices[i * self.global_batch_size : (i + 1) * self.global_batch_size]
                    for i in range(len(group_indices) // self.global_batch_size + self.drop_last)
                ]

        # Drop batches that don't divide evenly across GPUs
        n_leftover_indices = len(indices) % self.n_gpu
        if n_leftover_indices != 0:
            indices = indices[:-n_leftover_indices]

        # Shard indices across GPUs - each rank gets its own subset
        self.indices = indices[self.rank::self.n_gpu]

        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# ==================== DataLoader Builder ====================

def build_eeg_dataloader(
    args: EEGDataArgs,
    train: bool = True,
    state: Optional[EEGDataLoaderState] = None,
) -> Tuple[DataLoader, EEGDataLoaderState]:
    """
    Build EEG DataLoader compatible with ultra framework.

    Args:
        args: EEG data configuration
        train: Whether this is training (vs validation)
        state: Optional state for resuming

    Returns:
        dataloader: PyTorch DataLoader
        state: Current state (for checkpointing)
    """
    if state is None:
        state = EEGDataLoaderState()

    # Check for data in either 'recordings/' or 'data/' subdirectory
    recordings_candidate = pjoin(args.data_path, "recordings")
    data_candidate = pjoin(args.data_path, "data")
    if os.path.exists(recordings_candidate):
        recordings_path = recordings_candidate
    elif os.path.exists(data_candidate):
        recordings_path = data_candidate
    else:
        recordings_path = recordings_candidate  # fallback to original behavior

    csv_path = pjoin(args.data_path, "csv_recordings")

    # Read CSV metadata
    csv_big = _read_csv(pjoin(csv_path, "df_big.csv"))
    csv_corrected = _read_csv(pjoin(csv_path, "df_corrected.csv"))
    csv_stats = _read_csv(pjoin(csv_path, "df_stats_tmp.csv"))

    # Determine recording sets
    if args.subset == "small":
        train_set = SUBSET_TRAIN
        val_set = SUBSET_VAL
    elif args.subset == "open":
        train_set = OPEN_TRAIN
        val_set = OPEN_VAL
    elif args.subset == "all":
        train_set = list({int(row["big_recording_index"]) for row in csv_big})
        val_set = None
    else:
        raise ValueError(f"Unknown data subset: {args.subset}")

    # Filter data
    recording_set = train_set if train else (val_set if val_set else train_set)
    data = _filter_by_recording_set(csv_corrected, recording_set)
    data_big = _filter_by_recording_set(csv_big, recording_set)
    data_stats = _filter_by_recording_set(csv_stats, recording_set)

    # Filter out recordings where files don't exist
    existing_recordings = []
    for rec in data_big:
        r_i = int(rec["big_recording_index"])
        eeg_file = pjoin(recordings_path, f"recording_-_eeg_-_{r_i}.npy")
        if os.path.exists(eeg_file):
            existing_recordings.append(r_i)
        else:
            logger.warning(f"Skipping recording {r_i}: file not found")

    if not existing_recordings:
        raise ValueError(f"No valid EEG recordings found in {recordings_path}")

    logger.info(f"Found {len(existing_recordings)} valid recordings: {existing_recordings}")

    data = _filter_by_recording_set(data, existing_recordings)
    data_big = _filter_by_recording_set(data_big, existing_recordings)
    data_stats = _filter_by_recording_set(data_stats, existing_recordings)

    # Compute segments
    dict_groups, segments = compute_group_segments(data, data_big, args.window_duration)

    # Build dataset
    dataset = EEGDataset(
        segments, dict_groups, data_big, data_stats, recordings_path,
        window_duration=args.window_duration,
        clip=args.clip,
        block_masking=args.use_block_masking if train else False,
        masking_window=args.masking_window,
        masking_overlap=args.masking_overlap,
        masking_ratio=args.masking_ratio,
        radius_spat_mask=args.radius_spat_mask,
        radius_temp_mask=args.radius_temp_mask,
        dropout_ratio=args.dropout_ratio,
        dropout_radius=args.dropout_radius,
        no_masking=False,
        manual_seed=False if train else 42,
    )

    # Build sampler with rank for data sharding
    sampler = GroupedSampler(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        n_gpu=args.world_size,
        rank=args.rank,
        mode="train" if train else "val",
    )

    # Trigger sampler to generate indices
    _ = sampler.__iter__()

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    logger.info(f"Built EEG dataloader: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloader, state
