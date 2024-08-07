#iterationを指定する
ITERATION=10000

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
#


# distributed settings
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=4

FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/storage5/shared/Llama-3-35/Llama-3-35b-16nodes-tp4-pp4-ct1-LR1.0E-4-MINLR0.99E-4-WD0.1-WARMUP1000
HF_CHECKPOINT_DIR=/storage5/shared/Llama-3-35/HF/Llama-3-35b-16nodes-tp4-pp4-ct1-LR1.0E-4-MINLR0.99E-4-WD0.1-WARMUP1000/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

echo $ITERATION >"${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/storage5/shared/Llama-3-35/phase2_tokenizer_llama
cd /storage5/shared/hatakeyama/0611te/Megatron-LM
# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --megatron-path /storage5/shared/hatakeyama/0611te/Megatron-LM
