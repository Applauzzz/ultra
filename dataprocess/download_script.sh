#!/bin/bash
# $1: HF_DATASETS_CACHE
# $2: HUGGINGFACE_HUB_CACHE
# $3: repo_name
export HF_DATASETS_CACHE="$1"
export HUGGINGFACE_HUB_CACHE="$2"
python download_dataset.py --repo_name "togethercomputer/Long-Data-Collections"