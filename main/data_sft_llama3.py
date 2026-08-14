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
from typing import Dict, Any, Iterator, Optional, TypedDict, List, Union

from ultra.tokenizer import build_tokenizer, TokenizerArgs
# from .tokenizer import build_tokenizer
import numpy as np
import logging
import torch

"""
Llama-3 chat_template 参考：

"{% set loop_messages = messages %}{% for message in loop_messages %}
 {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'
                  + message['content'] | trim
                  + '<|eot_id|>' %}
 {% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}
 {{ content }}
{% endfor %}{% if add_generation_prompt %}
{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{% endif %}"

效果类似：

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Please answer in a concise way.<|eot_id|><|start_header_id|>user<|end_header_id|>

帮我用一句话介绍一下 Transformer 是什么？<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Transformer 是一种基于自注意力机制的序列建模架构，非常适合处理自然语言等序列数据。<|eot_id|>
"""

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


# ===================== SFT 类型定义 =====================

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


def tokenize_conversation_llama3(
    conversations: List[Dict[str, str]],
    tokenizer,
    add_bos: bool = True,
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
        text = (turn.get("content") or "").strip()

        # header：<|start_header_id|>role<|end_header_id|>\n\n
        header_text = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        header_tokens = tokenizer.encode(header_text, add_bos=False, add_eos=False)

        content_tokens = tokenizer.encode(text, add_bos=False, add_eos=False)

        # 第一个 message 加 bos_token
        if i == 0 and add_bos:
            bos_id = get_bos_id(tokenizer)
            input_ids.append(bos_id)
            labels.append(-100)  # BOS 不参与 loss

        if role == "assistant":
            # assistant：header mask，content + eot 参与监督
            input_ids.extend(header_tokens)
            labels.extend([-100] * len(header_tokens))

            input_ids.extend(content_tokens)
            labels.extend(content_tokens)

            input_ids.extend(eot_tokens)
            labels.extend(eot_tokens)
        else:
            # user / system：整条 message 只作为条件，全部 mask
            all_tokens = header_tokens + content_tokens + eot_tokens
            input_ids.extend(all_tokens)
            labels.extend([-100] * len(all_tokens))

        if len(input_ids) >= max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
            break

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels[-1] = labels[:max_seq_len]

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
        max_seq_len=args.max_seq_len,
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



def example_usage():
    sft_args = SFTDataArgs(
        sources={"load_dataset./home/zhliu/database/inf-ins/0625": 1.0},
        batch_size=8,
        max_seq_len=2048,
        tokenizer=TokenizerArgs(
            name="huggingface",  
            path="/home/zhliu/model_base/llama-3-8b", 
        ),
        # llama3没有自带的
        system_prompt="You are a helpful assistant. Please answer in a concise way.",
    )

    rank, world_size = 0, 1
    state = init_sft_dataloader_state(sft_args, rank, world_size)

    with build_sft_dataloader(state) as dataloader:
        for batch, new_state in dataloader:
            input_ids = torch.tensor(batch["input_ids"])
            labels = torch.tensor(batch["labels"])
            attention_mask = torch.tensor(batch["attention_mask"])

            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # loss = outputs.loss

            state = new_state
            tokenizer = build_tokenizer(name="huggingface", path="/home/zhliu/model_base/llama-3-8b")
            print("===== input_ids[0] =====")
            print(input_ids[0].tolist())  

            print("===== labels[0] =====")
            print(labels[0].tolist())

            print("text_decoded:")
            decoded_text = tokenizer.tokenizer.decode(
                input_ids[0],
                skip_special_tokens=False,
            )
            print(decoded_text)
            break

            
            # print(f"Processed batch with {len(input_ids)} samples, seq_len={input_ids.shape[1]}")

def example_usage_2():

    tokenizer = build_tokenizer(
        name="huggingface",
        path="/home/zhliu/model_base/llama-3-8b",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please answer in a concise way.",
        },
        {
            "role": "user",
            "content": "帮我用一句话介绍一下 Transformer 是什么？",
        },
        {
            "role": "assistant",
            "content": "Transformer 是一种基于自注意力机制的序列建模架构，非常适合处理自然语言等序列数据。",
        },
    ]
    # messages = [
    #     {
    #         "from": "human",
    #         "value": "In a certain country populated predominantly by wizards, there is an impending demonstration that needs careful planning to ensure it meets specific requirements.\n\nThere is a population of n individuals within the city, of which x are wizards committed to attending the demonstration. The remaining (n - x) individuals, non-wizards, have no intention to participate. The city administration will acknowledge and act upon the demonstration only if it engages at least y percent of the total city population. To meet this criterion, wizards are considering the creation of clone puppets that will stand in for the non-wizards at the demonstration.\n\nImportantly, the administration cannot distinguish between real individuals and the clone puppets when calculating participation percentages. They will treat the city's population as n for this calculation, completely unaware of the existence of any clones.\n\nYour task is to determine the minimum number of clone puppets the wizards must create to ensure that the demonstration involves no less than y percent of the original city population n.\n\nInput Format:\n\nThe first line of input includes three space-separated integers, n, x, y (1 ≤ n, x, y ≤ 104, x ≤ n) — representing the total number of city residents, the number of wizards, and the percentage required by the administration, respectively.\n\nIt should be noted that y can be a value exceeding 100 percent, meaning the administration might require a demonstration crowd size greater than the city’s actual population.\n\nOutput Format:\n\nPrint a single integer — the minimum number of clone puppets that need to be created to ensure the demonstration meets or exceeds y percent of n (the city's true population size).\n\nExamples:\n\nInput\n\n10 1 14\n\nOutput\n\n1\n\nInput\n\n20 10 50\n\nOutput\n\n0\n\nInput\n\n1000 352 146\n\nOutput\n\n1108\n\nNote:\n\nIn the first example, to meet the requirement that at least 14% of 10 people (which rounds up to 2) attend, the lone wizard would need to create one clone.\nIn the second example, all 10 required participants are wizards, so no clones are necessary.\nIn the third example, the requirement is for 1460 attendees, so with 352 wizards, an additional 1108 clone puppets are required."
    #     },
    #     {
    #         "from": "gpt",
    #         "value": "To determine the minimum number of clone puppets required for the demonstration, let's break down the problem step by step.\n\n### Problem Breakdown:\n1. We need to ensure that the total number of participants in the demonstration meets at least \\( y \\% \\) of the original city population \\( n \\).\n2. The wizards, \\( x \\) of them, will definitely participate.\n3. The administration does not want to know about the clone puppets; hence they treat the total city's population as \\( n \\).\n\n### Steps to Solve:\n1. **Calculate Requirement:** Determine the number of participants required to meet \\( y \\% \\) of \\( n \\). This can be done using:\n   \\[\n   \\text{required\\_participants} = \\left\\lceil \\frac{y}{100} \\times n \\right\\rceil\n   \\]\n   We need at least this number of participants in the demonstration.\n   \n2. **Compare with Existing Participants:** Check how many more participants are needed beyond the wizards available:\n   \\[\n   \\text{additional\\_participants\\_needed} = \\max(0, \\text{required\\_participants} - x)\n   \\]\n   If the wizards already meet or exceed the requirement, no additional clones are necessary. Otherwise, calculate the difference.\n\n### Formula Application Using Example:\n- **Example:** For \\( n = 10, x = 1, y = 14 \\):\n  1. Calculate required participants: \\( \\left\\lceil \\frac{14}{100} \\times 10 \\right\\rceil = \\left\\lceil 1.4 \\right\\rceil = 2 \\)\n  2. Calculate clones needed: \\( \\max(0, 2 - 1) = 1 \\)\n\n### Implementation:\nLet's implement this logic in a Python function:\n\n```python\nimport math\n\ndef minimum_clones_needed(n, x, y):\n    # Calculate the required number of participants\n    required_participants = math.ceil((y / 100) * n)\n    \n    # Calculate how many more participants are needed beyond the wizards\n    additional_needed = max(0, required_participants - x)\n    \n    return additional_needed\n\n# Read the input values\nn, x, y = map(int, input().split())\n\n# Calculate the minimum number of clone puppets required\nresult = minimum_clones_needed(n, x, y)\n\n# Print the result\nprint(result)\n```\n\n### Detailed Explanation:\n1. **Input Reading:** Read three integers \\( n, x, y \\).\n2. **Compute Required Participants:** Using \\(\\left\\lceil \\frac{y}{100} \\times n \\right\\rceil\\), calculate how many people need to be at the demonstration.\n3. **Determine Needed Clones:** Compare this number to the number of wizards \\( x \\). If \\( x \\) is less than the required number, calculate the difference; otherwise, zero clones are needed.\n4. **Output Result:** Print the minimum number of clone puppets needed.\n\nThis approach ensures that the demonstration meets the criterion set by the administration efficiently and correctly."
    #     },
    # ]
    messages = parse_conversation_format(messages)
 
    sft_sample = tokenize_conversation_llama3(
        conversations=messages,
        tokenizer=tokenizer,
        add_bos=True,
        max_seq_len=2048,
    )
    sft_target = tokenizer.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
    input_ids = sft_sample["input_ids"]
    labels = sft_sample["labels"]
    attention_mask = sft_sample["attention_mask"]

    processed_text = tokenizer.tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
    )

    print("===== sft预处理 =====")
    print(input_ids)
    print("length:", len(input_ids))
    print("===== sft target(apply chat template) =====")
    print(sft_target)
    print("length:",len(sft_target))
   
    print("\n===== input_ids =====")
    print(input_ids)
    print("\n===== labels =====")
    print(labels)
    print("\n===== attention_mask =====")
    print(attention_mask)

example_usage()
