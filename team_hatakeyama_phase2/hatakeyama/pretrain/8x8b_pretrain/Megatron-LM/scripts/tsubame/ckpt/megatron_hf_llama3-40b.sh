# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=4

ITERATION=6000
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/storage5/shared/Llama-3-35/Llama-3-35b-16nodes_2nd_tonyu-tp4-pp4-ct1-LR2E-5-MINLR1.99E-5-WD0.1-WARMUP8000
HF_CHECKPOINT_DIR=/storage5/shared/Llama-3-35/HF/Llama-3-35b-16nodes_2nd_tonyu-tp4-pp4-ct1-LR2E-5-MINLR1.99E-5-WD0.1-WARMUP8000/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

# echo $ITERATION >"${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/storage5/shared/Llama-3-35/phase2_tokenizer_llama

# convert
#python /storage5/shared/hatakeyama/0611te/Megatron-LM/tools/checkpoint/convert.py \
python /storage5/Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --megatron-path /storage5/Megatron-LM