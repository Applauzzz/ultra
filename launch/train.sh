export WANDB_OFFICIAL=1;
# export HF_DATASETS_CACHE="/home/zhliu/database/hf-cache";
# export HUGGINGFACE_HUB_CACHE="/home/zhliu/database/hub_cache";
export WANDB_API_KEY="95b93dfa45cbbfbcfdff982ea7d2cc8f6ef5a3b6";
# export PYTHONPATH=$(pwd)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TRITON_AUTOTUNE=0
export TRITON_ENABLE_AUTOTUNING=0
torchrun --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=29500 \
        ./main/train_sft.py config=./main/configs/qwen3_8B.yaml \
        2>&1 | tee /mnt/jfzn/lzh/output.log