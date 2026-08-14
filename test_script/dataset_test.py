import os
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import DeviceMesh

from streaming import StreamingDataset
from streaming import StreamingDataLoader

def setup_distributed():
    """
    Initializes the distributed process group.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def save_dataloader_state(dataloader, checkpoint_dir, rank):
    """Saves the DataLoader state for a specific rank."""
    # Ensure the checkpoint directory exists (only created by the main process).
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    # All processes wait for the main process to create the directory to ensure the path exists.
    dist.barrier()
    
    # Get the state dictionary of the dataloader.
    state_dict = dataloader.state_dict()
    
    # Construct a filename specific to this rank.
    save_path = os.path.join(checkpoint_dir, f"dataloader_rank_{rank}.pt")
    
    # Save the state.
    torch.save(state_dict, save_path)
    if rank == 0:
        print(f"Successfully saved DataLoader states to {checkpoint_dir}")

def load_dataloader_state(dataloader, checkpoint_dir, rank):
    """Loads the DataLoader state for a specific rank."""
    load_path = os.path.join(checkpoint_dir, f"dataloader_rank_{rank}.pt")
    if os.path.exists(load_path):
        # Load to CPU first to avoid potential GPU memory issues.
        state_dict = torch.load(load_path, map_location='cpu') 
        dataloader.load_state_dict(state_dict)
        if rank == 0:
            print(f"Successfully loaded DataLoader states from {checkpoint_dir}")
        return True
    else:
        if rank == 0:
            print(f"Warning: No DataLoader state found at {load_path}. Starting from scratch.")
        return False

if __name__ == "__main__":
    rank, world_size, local_rank = setup_distributed()
    
    dataset = StreamingDataset(
        local="/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/data/mds_fineweb_edu",
        batch_size=1,
        shuffle=True,
        shuffle_seed=42,
    )
    
    dataloader = StreamingDataLoader(dataset, batch_size=1)
    
    concate = ""
    count = 0
    
    load_dataloader_state(dataloader, "./ckpt/", rank)

    import pdb
    print(len(dataloader))
    if dist.get_rank() == 0:
        pdb.set_trace()
    dist.barrier()
    
    for data in dataloader:
        strs = data["text"][0][:10]
        print(f"{rank}: {strs}")
        exit()

        # if count == 10:
        #     print(f"{rank}: {strs}")
        # else:
        #     if count == 20:
        #         exit()
        # count += 1
