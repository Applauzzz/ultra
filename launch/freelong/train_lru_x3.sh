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
        --master_port=29503 \
        ./main/train_lru.py config=./main/model_config/lru/lru.yaml\
        2>&1 | tee /mnt/zehao/output_lru_3.log