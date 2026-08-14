from ultra.mds_data import build_dataloader
from streaming import StreamingDataset, StreamingDataLoader
from transformers import AutoTokenizer

if __name__ == "__main__":
    dataset = StreamingDataset(
        local="/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/data/mds_fineweb_edu",
        batch_size=1,
    )
    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/model/llama-2-7b")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=0,
        world_size=1,
        batch_size=32,
        seq_len=2048,
        context_len=2048,
        varlen=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        snapshot_every_n_steps=1,
    )
    count = 0
    for batch in dataloader:
        
        count += 1
        if count == 9:
            print(batch)
            state = dataloader.state_dict()
        if count == 10:
            print(batch)
            break
    dataset = StreamingDataset(
        local="/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/data/mds_fineweb_edu",
        batch_size=1,
    )
    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/model/llama-2-7b")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=0,
        world_size=1,
        batch_size=32,
        seq_len=2048,
        context_len=2048,
        varlen=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        snapshot_every_n_steps=1,
    )
    dataloader.load_state_dict(state)
    for batch in dataloader:
        print(batch)
        break
    
