
export HF_DATASETS_CACHE="$1";
export HUGGINGFACE_HUB_CACHE="$2";

python main/harness_eval.py \
    --model lingua \
    --model_args pretrained=$3,model_name=transformer,tokenizer=$4,max_length=16384 \
    --tasks based_triviaqa \
    --device cuda:0 \
    --num_fewshot 0 \
    --batch_size 16