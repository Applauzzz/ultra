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

torchrun --nproc_per_node=8 \
        --nnodes=4 \
        --node_rank=2 \
        --master_addr=10.60.157.155 \
        --master_port=29501 \
        ./main/train.py config=/mnt/zehao/ULTra/main/model_config/nemo/swagdn_1B_256k.yaml \
        # 2>&1 | tee /mnt/zehao/swagdn_1B.log

# conda activate liu
# cd /mnt/zehao/ULTra
# export PYTHONPATH=$(pwd)

# bash /mnt/zehao/ULTra/launch/train_nemo_1.sh