#!/bin/bash
# $1: HF_DATASETS_CACHE
# $2: HUGGINGFACE_HUB_CACHE
# $3: output_dir
export HF_DATASETS_CACHE="$1"
export HUGGINGFACE_HUB_CACHE="$2"
python hf_dataset_to_jsonl.py \
    --repo_name "togethercomputer/Long-Data-Collections" \
    --out_dir "$3" \
    --num_chunks 64