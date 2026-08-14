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
    DataArgs,
    async_iterator,
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
    batch_size: int
    system_prompt: Optional[str]


# ===================== Llama-3 专用 helper =====================

def normalize_role_llama3(role: str) -> str:
    """
    统一 role 名称，Llama-3 期望：system / user / assistant
    """
    r = role.lower()
    if r == "system":
        return "system"
    if r in ["human", "user"]:
        return "user"
    if r in ["assistant", "bot", "gpt"]:
        return "assistant"
    return "user"


def get_bos_id(tokenizer, default: int = 1) -> int:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        return default
    return bos_id


def get_eot_tokens(tokenizer) -> List[int]:

    try:
        tokens = tokenizer.encode("<|eot_id|>", add_bos=False, add_eos=False)
        if len(tokens) > 0:
            return tokens
    except Exception:
        pass

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return [eos_id]

    logging.warning("Tokenizer has no <|eot_id|> or eos_token_id, using '\\n' as EOT token.")
    return tokenizer.encode("\n", add_bos=False, add_eos=False)


def get_system_prompt_text(
    tokenizer,
    override: Optional[str] = None,
) -> Optional[str]:

    if override is not None:
        return override

    for attr in ["default_system_prompt", "system_prompt"]:
        text = getattr(tokenizer, attr, None)
        if isinstance(text, str) and text.strip():
            return text

    return None


def parse_conversation_format(content: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:


    # {"conversations": [{"role": "human", "content": "..."}]}
    if "conversations" in content:
        convs = []
        for msg in content["conversations"]:
            role = msg.get("role") or msg.get("from") or "user"
            text = msg.get("content") or msg.get("value") or ""
            convs.append({"role": role, "content": text})
        return convs

    # {"messages": [{"role": "user", "content": "..."}]}
    elif "messages" in content:
        convs = []
        for msg in content["messages"]:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            convs.append({"role": role, "content": text})
        return convs

    # {"instruction": "...", "input": "...", "output": "..."}
    elif "instruction" in content and "output" in content:
        convs = []
        human_content = content["instruction"]
        if content.get("input", "").strip():
            human_content += f"{content['input']}"
        convs.append({"role": "user", "content": human_content})
        convs.append({"role": "assistant", "content": content["output"]})
        return convs

    # {"prompt": "...", "response": "..."}
    elif "prompt" in content and "response" in content:
        return [
            {"role": "user", "content": content["prompt"]},
            {"role": "assistant", "content": content["response"]},
        ]

    return None


def normalize_content_to_str(content: Any) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return str(content)

    # 其他类型兜底
    return str(content)

def tokenize_conversation_llama3(
    conversations: List[Dict[str, Any]],
    tokenizer,
    add_bos: bool = False,
    max_seq_len: int = 2048,
) -> Optional[SFTSample]:

    if not conversations:
        return None

    input_ids: List[int] = []
    labels: List[int] = []

    eot_tokens = get_eot_tokens(tokenizer)

    for i, turn in enumerate(conversations):
        raw_role = turn.get("role", "user")
        role = normalize_role_llama3(raw_role)

        raw_content = turn.get("content")
        text = normalize_content_to_str(raw_content).strip()

        header_text = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        header_tokens = tokenizer.encode(header_text, add_bos=False, add_eos=False)

        content_tokens = tokenizer.encode(text, add_bos=False, add_eos=False)

        if i == 0 and add_bos:
            bos_id = get_bos_id(tokenizer)
            input_ids.append(bos_id)
            labels.append(-100)  # BOS 不参与 loss

        if role == "assistant":
            input_ids.extend(header_tokens)
            labels.extend([-100] * len(header_tokens))

            input_ids.extend(content_tokens)
            labels.extend(content_tokens)

            input_ids.extend(eot_tokens)
            labels.extend(eot_tokens)
        else:
            all_tokens = header_tokens + content_tokens + eot_tokens
            input_ids.extend(all_tokens)
            labels.extend([-100] * len(all_tokens))

        if len(input_ids) >= max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
            break

    attention_mask = [1] * len(input_ids)

    return SFTSample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

def sft_tokenize(
    iterator,
    add_bos: bool,
    add_eos_ignored: bool,  # 对齐原接口，但 Llama-3 模板本身不再单独加 EOS
    tokenizer_type: str,
    tokenizer_path: Optional[str] = None,
    max_seq_len: int = 2048,
    system_prompt: Optional[str] = None,
):

    tokenizer = build_tokenizer(name=tokenizer_type, path=tokenizer_path)
    system_prompt_text = get_system_prompt_text(tokenizer, override=system_prompt)

    for content, state in iterator:
        conversations = parse_conversation_format(content)
        if conversations is None:
            continue

        if system_prompt_text:
            has_system = any(
                normalize_role_llama3(m.get("role", "user")) == "system"
                for m in conversations
            )
            if not has_system:
                conversations = [{"role": "system", "content": system_prompt_text}] + conversations

        sft_sample = tokenize_conversation_llama3(
            conversations,
            tokenizer,
            add_bos=add_bos,
            max_seq_len=max_seq_len,
        )

        if sft_sample is not None:
            # 至少有一个 label != -100 才保留样本
            has_valid_label = any(label != -100 for label in sft_sample["labels"])
            if not has_valid_label:
                continue

            yield sft_sample, TokenizerState(
                it_state=state,
                add_bos=add_bos,
                add_eos=add_eos_ignored,
                name=tokenizer_type,
                path=tokenizer_path,
            )


# ===================== pack & pad =====================

def sft_pack_samples(
    iterator,
    empty_buffer_state: SFTPackState,
):
    buffer = empty_buffer_state["buffer"]
    max_seq_len = empty_buffer_state["max_seq_len"]
    pad_token_id = empty_buffer_state["pad_token_id"]
    previous_state = empty_buffer_state["it_state"]
    batch_size = empty_buffer_state.get("batch_size", 1)
    system_prompt = empty_buffer_state.get("system_prompt")

    for sft_sample, state in iterator:
        buffer.append(sft_sample)

        if len(buffer) >= batch_size:
            batch = pad_sft_batch(buffer, max_seq_len, pad_token_id)
            new_state: SFTPackState = SFTPackState(
                buffer=[],
                it_state=state,
                max_seq_len=max_seq_len,
                pad_token_id=pad_token_id,
                batch_size=batch_size,
                system_prompt=system_prompt,
            )
            yield batch, new_state
            buffer = []
            previous_state = state


def pad_sft_batch(
    samples: List[SFTSample],
    max_seq_len: int,
    pad_token_id: int,
) -> Dict[str, np.ndarray]:
    batch_size = len(samples)
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
        "attention_mask": attention_mask,
    }

@dataclass
class SFTDataArgs:
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 8
    max_seq_len: int = 2048
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = False 
    load_async: bool = True
    prefetch_size: int = 16
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    pad_token_id: int = 128255  # Llama-3  pad_token_id !!!!!!!!!!!!!!
    system_prompt: Optional[str] = None  # llama3 需要手动 sysprompt 文本


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
        file_pattern="*.jsonl",
    )

    tokenizer_state = TokenizerState(
        it_state=multi_choice_state,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        name=args.tokenizer.name,
        path=args.tokenizer.path,
    )

    sft_pack_state: SFTPackState = SFTPackState(
        buffer=[],
        it_state=tokenizer_state,
        max_seq_len=args.seq_len,
        pad_token_id=args.pad_token_id,
        batch_size=args.batch_size,
        system_prompt=args.system_prompt,
    )

    return PrefetchState(
        it_state=sft_pack_state,
        seq_idx=0,
        rng_state=np.random.default_rng((args.seed, rank, world_size)).bit_generator.state,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
    )


@contextlib.contextmanager
def build_sft_dataloader(
    state: PrefetchState,
):
    sft_pack_state: SFTPackState = state["it_state"]
    print(sft_pack_state)
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
        tokenizer_state["add_eos"],  # 在 Llama-3 分支里，这个参数只用于保留接口，不再加 EOS
        tokenizer_state["name"],
        tokenizer_state["path"],
        sft_pack_state["max_seq_len"],
        system_prompt=sft_pack_state.get("system_prompt"),
    )

    data_it = sft_pack_samples(data_it, sft_pack_state)

    try:
        yield data_it
    finally:
        for it in path_to_iter.values():
            it.close()
        data_it.close()



def build_sft_dataloader_from_args(
    args: DataArgs,
    state: Optional[PrefetchState] = None,
):
    data_builder = partial(build_sft_dataloader, state)
    if args.load_async:
        return async_iterator(args.prefetch_size, data_builder)
    else:
        return data_builder()