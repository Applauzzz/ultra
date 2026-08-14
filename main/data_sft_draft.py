import contextlib
from copy import deepcopy
from functools import partial
import json
from dataclasses import dataclass, field
from datasets import load_from_disk, load_dataset
from datasets import Dataset
from multiprocessing import Process, Queue, Event
from queue import Full, Empty
from multiprocessing.synchronize import Event as EventClass
import os
from pathlib import Path
from queue import Full
from typing import Dict, Any, Iterator, Optional, TypedDict
from ultra.tokenizer import build_tokenizer, TokenizerArgs

# from .tokenizer import build_tokenizer
import numpy as np
import logging
import torch
from typing import List, Union

from ultra.data import (
    PrefetchState,
    TokenizerState,
    init_choice_state,
    choose_source,
    setup_sources,
    build_tokenizer,
    TokenizerArgs,
)


class SFTSample(TypedDict):
    input_ids: List[int]     
    labels: List[int]      
    attention_mask: List[int] 
    
class SFTPackState(TypedDict):
    buffer: List[SFTSample]   
    it_state: Any       
    max_seq_len: int      
    pad_token_id: int   
    

def parse_conversation_format(content: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:

    # {"conversations": [{"role": "human", "content": "..."}]} 
    if "conversations" in content:
        return content["conversations"]
    
    # {"messages": [{"role": "user", "content": "..."}]}
    elif "messages" in content:
        conversations = []
        for msg in content["messages"]:
            role = "human" if msg["role"] in ["user", "human"] else "assistant"
            conversations.append({"role": role, "content": msg["content"]})
        return conversations
    
    # {"instruction": "...", "input": "...", "output": "..."}
    elif "instruction" in content and "output" in content:
        conversations = []
        human_content = content["instruction"]
        if content.get("input", "").strip():
            human_content += f"\n\n{content['input']}"
        
        conversations.append({"role": "human", "content": human_content})
        conversations.append({"role": "assistant", "content": content["output"]})
        return conversations
    
    # {"prompt": "...", "response": "..."}
    elif "prompt" in content and "response" in content:
        return [
            {"role": "human", "content": content["prompt"]},
            {"role": "assistant", "content": content["response"]}
        ]
    
    return None

def tokenize_conversation(
    conversations: List[Dict[str, str]], 
    tokenizer,
    add_bos: bool = True,
    add_eos: bool = True,
    max_seq_len: int = 2048
) -> Optional[SFTSample]:

    if not conversations:
        return None
    
    input_ids = []
    labels = []

    if add_bos:
        bos_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1
        input_ids.append(bos_id)
        labels.append(-100)  # BOS不参与loss计算
    
    for i, turn in enumerate(conversations):
        role = turn["from"]
        content = turn["value"]
        if role == "human":
            role_prefix = "Human: "
        else:
            role_prefix = "Assistant: "

        role_tokens = tokenizer.encode(role_prefix, add_bos=False, add_eos=False)
        content_tokens = tokenizer.encode(content, add_bos=False, add_eos=False)
        
        if role == "human":
            input_ids.extend(role_tokens + content_tokens)
            labels.extend([-100] * (len(role_tokens) + len(content_tokens)))
        else:
            input_ids.extend(role_tokens + content_tokens)
            labels.extend([-100] * len(role_tokens) + content_tokens) 

        sep_token = tokenizer.encode("\n", add_bos=False, add_eos=False)
        input_ids.extend(sep_token)
        if role == "human":
            labels.extend([-100] * len(sep_token))
        else:
            labels.extend(sep_token)
    
    if add_eos:
        eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2
        input_ids.append(eos_id)
        labels.append(eos_id)
    
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
    
    attention_mask = [1] * len(input_ids)
    
    return SFTSample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask
    )

def sft_tokenize(
    iterator,
    add_bos: bool,
    add_eos: bool,
    tokenizer_type: str,
    tokenizer_path: Optional[str] = None,
    max_seq_len: int = 2048,
):
    tokenizer = build_tokenizer(name=tokenizer_type, path=tokenizer_path)
    
    for content, state in iterator:
        conversations = parse_conversation_format(content)
        if conversations is None:
            continue 

        sft_sample = tokenize_conversation(
            conversations, 
            tokenizer, 
            add_bos, 
            add_eos, 
            max_seq_len
        )
        
        if sft_sample is not None:
            yield sft_sample, TokenizerState(
                it_state=state,
                add_bos=add_bos,
                add_eos=add_eos,
                name=tokenizer_type,
                path=tokenizer_path,
            )

def sft_pack_samples(
    iterator,
    empty_buffer_state: SFTPackState,
):
    buffer = empty_buffer_state["buffer"]
    max_seq_len = empty_buffer_state["max_seq_len"]
    pad_token_id = empty_buffer_state["pad_token_id"]
    previous_state = empty_buffer_state["it_state"]
    
    for sft_sample, state in iterator:
        buffer.append(sft_sample)
        
        if len(buffer) >= empty_buffer_state.get("batch_size", 1):
            batch = pad_sft_batch(buffer, max_seq_len, pad_token_id)
            
            new_state = SFTPackState(
                buffer=[],
                it_state=state,
                max_seq_len=max_seq_len,
                pad_token_id=pad_token_id,
            )
            
            yield batch, new_state
            buffer = []
            previous_state = state

def pad_sft_batch(samples: List[SFTSample], max_seq_len: int, pad_token_id: int) -> Dict[str, np.ndarray]:

    batch_size = len(samples)
    
    # max_len = max(len(sample["input_ids"]) for sample in samples)
    max_len = max_seq_len

    input_ids = np.full((batch_size, max_len), pad_token_id, dtype=np.int64)
    labels = np.full((batch_size, max_len), -100, dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
    
    for i, sample in enumerate(samples):
        seq_len = min(len(sample["input_ids"]), max_len)
        
        input_ids[i, :seq_len] = sample["input_ids"][:seq_len]
        labels[i, :seq_len] = sample["labels"][:seq_len]
        attention_mask[i, :seq_len] = sample["attention_mask"][:seq_len]
    
    return {
        "input_ids": input_ids,
        "labels": labels, 
        "attention_mask": attention_mask
    }

@dataclass
class SFTDataArgs:
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 8
    max_seq_len: int = 2048
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 16  # SFT通常prefetch更少
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    pad_token_id: int = 0
    
def init_sft_dataloader_state(
    args: SFTDataArgs,
    rank: int,
    world_size: int,
):

    multi_choice_state = init_choice_state(
        root_dir=args.root_dir,
        sources=args.sources,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        file_pattern="*.jsonl" 
    )
    
    tokenizer_state = TokenizerState(
        it_state=multi_choice_state,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        name=args.tokenizer.name,
        path=args.tokenizer.path,
    )
    
    sft_pack_state = SFTPackState(
        buffer=[],
        it_state=tokenizer_state,
        max_seq_len=args.max_seq_len,
        pad_token_id=args.pad_token_id,
        batch_size=args.batch_size,
    )
    
    return PrefetchState(
        it_state=sft_pack_state,
        seq_idx=0,
        rng_state=np.random.default_rng((args.seed, rank, world_size)).bit_generator.state,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
    )

@contextlib.contextmanager
def build_sft_dataloader(state: PrefetchState):
    sft_pack_state = state["it_state"]
    tokenizer_state = sft_pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]
    
    path_to_iter = setup_sources(multi_state)
    
    data_it = choose_source(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
    )
    
    data_it = sft_tokenize(
        data_it,
        tokenizer_state["add_bos"],
        tokenizer_state["add_eos"],
        tokenizer_state["name"],
        tokenizer_state["path"],
        sft_pack_state["max_seq_len"],
    )
    
    data_it = sft_pack_samples(data_it, sft_pack_state)
    
    try:
        yield data_it
    finally:
        for it in path_to_iter.values():
            it.close()
        data_it.close()


def example_usage():
    sft_args = SFTDataArgs(
        sources={"load_from_disk./mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/data/Infinity_Instruct": 1.0},
        batch_size=128,
        max_seq_len=2048,
        tokenizer=TokenizerArgs(name="huggingface", path="/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/model/Mistral-7B-v0.1"),
    )
    
    rank, world_size = 0, 1 
    state = init_sft_dataloader_state(sft_args, rank, world_size)
    
    with build_sft_dataloader(state) as dataloader:
        count = 0
        for batch, new_state in dataloader:

            input_ids = torch.tensor(batch["input_ids"])
            labels = torch.tensor(batch["labels"])
            attention_mask = torch.tensor(batch["attention_mask"])
            
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # loss = outputs.loss  

            count += 1
            if count >= 100_000:
                break
            state = new_state
            # print(f"Processed batch with {len(input_ids)} samples, seq_len={input_ids.shape[1]}")
            print(count)

example_usage()