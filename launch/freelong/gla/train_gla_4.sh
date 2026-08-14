export WANDB_OFFICIAL=1;
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HF_DATASETS_CACHE="/home/zhliu/database/hf-cache";
# export HUGGINGFACE_HUB_CACHE="/home/zhliu/database/hub_cache";
export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6";
# export PYTHONPATH=$(pwd)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=29501 \
        ./main/train_freelong.py config=/mnt/zehao/ULTra/main/model_config/gla/gla_7B_8k.yaml\
        # 2>&1 | tee /mnt/zehao/output_gla_ori.log


# set -euo pipefail

# export WANDB_OFFICIAL=1
# export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6"
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TRITON_AUTOTUNE=0
# export TRITON_ENABLE_AUTOTUNING=0

# BASE_CONFIG=/mnt/zehao/ULTra/main/model_config/hgrn/hgrn.yaml
# TMP_CONFIG_DIR=/mnt/zehao/ULTra/main/model_config/tmp_configs
# LOG_DIR=/mnt/zehao/ULTra/main/model_log/log
# CKPT_ROOT=/mnt/zehao/ULTra/main/model_ckpt/ckpt

# mkdir -p "$TMP_CONFIG_DIR" "$LOG_DIR"

# # ===== 起点：1M =====
# seq_len=128
# seq_len=1048576   # 1M
# max_seq_len=1073741824  # 1B

# while [ "$seq_len" -le "$max_seq_len" ]; do

#     # ===== 生成可读 tag =====
#     if [ "$seq_len" -ge 1073741824 ]; then
#         seq_tag="$((seq_len / 1024 / 1024 / 1024))b"
#     elif [ "$seq_len" -ge 1048576 ]; then
#         seq_tag="$((seq_len / 1024 / 1024))m"
#     else
#         seq_tag="$((seq_len / 1024))k"
#     fi

#     dump_dir="${CKPT_ROOT}/hgrn-${seq_tag}"
#     run_name="hgrn-${seq_tag}"
#     tmp_config="${TMP_CONFIG_DIR}/gru_${seq_tag}.yaml"
#     log_file="${LOG_DIR}/output_freelong_${run_name}.log"

#     mkdir -p "$dump_dir"

#     echo "======================================"
#     echo "Running seq_len=${seq_len} (${seq_tag})"
#     echo "======================================"

#     python /mnt/zehao/ULTra/main/model_config/gru/rewrite_yaml.py \
#         --input "$BASE_CONFIG" \
#         --output "$tmp_config" \
#         --seq_len "$seq_len" \
#         --dump_dir "$dump_dir" \
#         --name "$run_name"

#     torchrun --nproc_per_node=1 \
#         ./main/train_lru.py \
#         config="$tmp_config" \
#         2>&1 | tee "$log_file"

#     # ===== 4倍增长 =====
#     seq_len=$((seq_len * 4))

# done