
torchrun --nproc_per_node=8 \
        --nnodes=$ARNOLD_WORKER_NUM \
        --node_rank=$ARNOLD_ID \
        --master_addr=$ARNOLD_WORKER_0_HOST \
        --master_port=${ARNOLD_WORKER_0_PORT%%,*} \
        ./test_script/model_to_ulysses.py