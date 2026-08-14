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

# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/gla_7B_96k.yaml \
#         2>&1 | tee /mnt/jfzn/lzh/output_freelong_7B_96k.log
# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_128k.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_128k.log
# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_512k.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_512k.log

# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_1M.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_1M.log


# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_2M.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_2M.log

# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_4M.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_4M.log

# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_8M.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_8M.log

# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29500 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_16M.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_16M.log

# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29503 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_128M.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_128M.log

# torchrun --nproc_per_node=8 \
#         --nnodes=1 \
#         --node_rank=0 \
#         --master_addr=127.0.0.1 \
#         --master_port=29503 \
#         ./main/train.py config=./main/configs/act_fsdp2_sp4/gla_7B_256M.yaml \
#         2>&1 | tee /mnt/zehao/output_freelong_7B_256M.log