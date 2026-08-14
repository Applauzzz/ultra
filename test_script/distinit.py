import os
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, distribute_tensor
from torch.distributed._tensor.placement_types import Shard, Replicate

def setup_distributed():
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

if __name__ == "__main__":
    
    rank, world_size, local_rank = setup_distributed()
    mesh = DeviceMesh("cuda", list(range(world_size)))

    with torch.device("meta"):
        meta_tensor = torch.empty(8, 8)

    dtensor = distribute_tensor(meta_tensor, mesh, [Shard(0)])
    # dtensor = distribute_tensor(meta_tensor, mesh, [Replicate()])
    dtensor = nn.Parameter(dtensor)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        empty_dtensor = torch.empty_like(dtensor, device="cuda")
        nn.init.kaiming_uniform_(empty_dtensor)


    print(type(empty_dtensor))
    print(isinstance(empty_dtensor, DTensor))
    print(empty_dtensor)
    dist.destroy_process_group()
