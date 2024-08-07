#!/bin/bash
source /storage5/shared/abeja/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_abeja_py310_TEv1.7_FAv2.5.7

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export NVTE_APPLY_QK_LAYER_SCALING=1

# distributed settings
MASTER_ADDR=slurm0-a3-ghpc-1
MASTER_PORT=65001
NODE_RANK=${1}
echo "Node rank: "$NODE_RANK
#NODE_RANK=0
NNODES=2
GPUS_PER_NODE=8

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8

# model config
# mixtral-8x7B-v0.1: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
model_size=8
NUM_LAYERS=32

HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # hiddensizeの3.5倍
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8

SEQ_LENGTH=2048

NUM_EXPERTS=4
NUM_EXPERT_TOP_K=2

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=16
CONTEXT_PARALLEL_SIZE=1
#DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=1.0E-5
MIN_LR=1.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"
CHECKPOINT_DIR="/storage5/shared/hatakeyama/0706moe_abeja/mergekit/megatron/model_four"
CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-8-8MoE/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

CACHE_DIR="/storage5/s_hared/Nishijima/cache"

TRAIN_DATA_PATH=""
#TRAIN_DATA_PATH="/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/0619synth_gcp_needed_clean_text_document"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3052499 /storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/openmath_text_document"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 8505192 /storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/synthtext1_text_document"
DATA_PATH_LIST=(
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/openmath_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/synthtext1_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/0619synth_gcp_needed_clean_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/0619synth_hf_needed_clean_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/sansu_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/synthtext0_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/wiki0_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/9_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/10_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/8_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/4_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/7_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/11_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/2_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/6_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/14_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/12_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/3_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/13_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/llmjp/5_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-47_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-14_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-30_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-36_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-22_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-51_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-35_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-05_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-49_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-43_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2013-20_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-18_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-26_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-17_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-10_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-15_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-17_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-35_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-30_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-22_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2020-16_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-09_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-23_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-26_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-07_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-40_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-11_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-09_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-43_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-18_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-13_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-43_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-04_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-13_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-26_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-52_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-50_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-44_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-34_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-04_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-40_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-22_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-48_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-22_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2016-22_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-18_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-47_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-35_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-09_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-39_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-26_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-51_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-42_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2014-41_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-30_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-39_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-27_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2013-48_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-30_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2018-39_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-32_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-34_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2015-06_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2019-47_text_document"
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2017-13_text_document"
)

# 配列をスペース区切りの文字列に変換
# TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")

# 配列をスペース区切りの文字列に変換
#TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
mkdir -p ${CHECKPOINT_SAVE_DIR}


#finetune
# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

JOB_NAME="Llama-3-8b-MoE_Nishijima_test"

# run
megatron_options="  \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_KEY_VALUE_HEADS} \
  --num-experts ${NUM_EXPERTS} \
  --router-aux-loss-coef 0.02 \
  --moe-type mixtral \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 998,1,1 \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 100 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --disable-bias-linear \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --log-throughput \
  --wandb-exp-name ${JOB_NAME} \
  --wandb-project "Llama-3-8B-MoE" \
  --wandb-entity "weblab-geniac1" \
  --router-aux-loss-coef 0.02 \
  --data-cache-path ${CACHE_DIR}
	"

#  --no-fp8-wgrad \ \\  --use-mcore-models \  --use-checkpoint-args \

current_time=$(date "+%Y.%m.%d_%H.%M.%S")

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#run_cmd="torchrun $DISTRIBUTED_ARGS /home/ext_yokobase_gmail_com/ABEJA/Megatron-LM/pretrain_gpt.py
#  ${megatron_options}"
run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/shared/abeja/Megatron-LM/pretrain_gpt.py
  ${megatron_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
exit 0
