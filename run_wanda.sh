MODEL="llama2-7b-chat-hf"
METHOD="wanda"
PRUNE_DATA="align"
OUTPUT_DIR="results/fig2a/wanda_0.01_2"
TYPE="unstructured"

python main.py \
    --model $MODEL \
    --prune_method $METHOD \
    --prune_data $PRUNE_DATA \
    --sparsity_ratio 0.01 \
    --sparsity_type $TYPE \
    --neg_prune \
    --save $OUTPUT_DIR \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res

python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

rm -rf temp/_vllm_tmp temp/tmp_vllm_model


