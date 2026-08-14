#!/bin/bash
# $1: local dir
python snapshot_dataset.py --repo_name "togethercomputer/Long-Data-Collections" \
                            --local_dir "$1"