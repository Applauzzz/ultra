import os
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import DeviceMesh

from streaming import StreamingDataset
from streaming import StreamingDataLoader

import pdb

if __name__ == "__main__":
    dataset = StreamingDataset(
        local="/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/data/mds_fineweb_edu",
        batch_size=1)
    dataloader = StreamingDataLoader(dataset, batch_size=1)
    for data in dataloader:
        pdb.set_trace()