#!/usr/bin/env bash
# Chunk-size ablation: hold seq_len = 128K, sweep FreeLong chunk_size from 8K up.
# Each run reuses the same base YAML and only differs in model.chunk_size.

export WANDB_OFFICIAL=1
export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0

# ===== NCCL / RDMA =====
unset NCCL_IB_DISABLE
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth1
export GLOO_SOCKET_IFNAME=eth1

# ===== paths (current cluster) =====
BASE_CONFIG=/mnt/zehao/ULTra/main/configs/rebuttal/gla_7B_chunk_ab,yaml
TMP_CONFIG_DIR=/mnt/zehao/ULTra/main/model_config/tmp_configs
LOG_DIR=/mnt/zehao/ULTra/main/model_log/log_chunkablation
CKPT_ROOT=/mnt/zehao/ULTra/ckpt/Ultra-gla
REWRITE_YAML=/mnt/zehao/ULTra/main/model_config/gru/rewrite_yaml.py
TRAIN_PY=/mnt/zehao/ULTra/main/train_freelong.py

mkdir -p "$TMP_CONFIG_DIR" "$LOG_DIR" "$CKPT_ROOT"

# ===== fixed total sequence length: 128K =====
SEQ_LEN=1048576

# ===== chunk sizes to sweep (≤ SEQ_LEN) =====
# 8K, 16K, 32K, 64K, 128K (= seq_len, i.e. single chunk / equivalent to plain BP)
CHUNK_SIZES=(8192 16384 32768 65536 131072 262144 524288 1048576)
# CHUNK_SIZES=(8192)
# ===== multi-node coords =====
NNODES=1
NPROC_PER_NODE=8
MASTER_ADDR=10.60.157.155
MASTER_PORT=29503

LOCAL_IP=$(hostname -I | awk '{print $1}')
case "$LOCAL_IP" in
  10.60.157.155) NODE_RANK=0 ;;
  10.60.35.33)   NODE_RANK=1 ;;
  10.60.233.146) NODE_RANK=0 ;;
  10.60.137.131) NODE_RANK=0 ;;
  *)
    echo "Unknown machine IP: $LOCAL_IP — assuming NODE_RANK=0"
    NODE_RANK=0
    ;;
esac

echo "LOCAL_IP=$LOCAL_IP  NODE_RANK=$NODE_RANK"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "SEQ_LEN=$SEQ_LEN"
echo "Chunk sizes to ablate: ${CHUNK_SIZES[*]}"
echo

_fmt_size() {
  # convert bytes-like int to "8k" / "32k" / "1m"
  local n=$1
  if   [ "$n" -ge 1073741824 ]; then echo "$((n / 1024 / 1024 / 1024))b"
  elif [ "$n" -ge 1048576 ];    then echo "$((n / 1024 / 1024))m"
  else                               echo "$((n / 1024))k"
  fi
}

seq_tag=$(_fmt_size "$SEQ_LEN")

for chunk_size in "${CHUNK_SIZES[@]}"; do
  if [ "$chunk_size" -gt "$SEQ_LEN" ]; then
    echo "[skip] chunk_size=$chunk_size > seq_len=$SEQ_LEN"
    continue
  fi

  chunk_tag=$(_fmt_size "$chunk_size")
  run_name="freelong-gla-seq${seq_tag}-chunk${chunk_tag}"
  dump_dir="${CKPT_ROOT}/gla-${run_name}"
  tmp_config="${TMP_CONFIG_DIR}/gla_seq${seq_tag}_chunk${chunk_tag}.yaml"
  log_file="${LOG_DIR}/output_${run_name}.log"

  mkdir -p "$dump_dir"

  echo "============================================================"
  echo "Running: seq_len=${SEQ_LEN} (${seq_tag})  chunk_size=${chunk_size} (${chunk_tag})"
  echo "  config -> $tmp_config"
  echo "  dump   -> $dump_dir"
  echo "  log    -> $log_file"
  echo "============================================================"

  python "$REWRITE_YAML" \
      --input  "$BASE_CONFIG" \
      --output "$tmp_config" \
      --seq_len "$SEQ_LEN" \
      --chunk_size "$chunk_size" \
      --dump_dir "$dump_dir" \
      --name    "$run_name"

  torchrun \
      --nnodes=1 \
      --nproc_per_node=8 \
      --master_port="$MASTER_PORT" \
      "$TRAIN_PY" \
      config="$tmp_config" \
      2>&1 | tee "$log_file"
  echo
done
