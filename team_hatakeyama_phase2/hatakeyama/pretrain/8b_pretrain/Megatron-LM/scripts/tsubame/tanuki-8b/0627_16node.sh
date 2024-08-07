#!/bin/bash

#checkpoint conversion
#/storage5/shared/hatakeyama/0611te/Megatron-LM/scripts/tsubame/ckpt/hf_to_megatron_llama3-8b_0627.sh

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#**********************
#マスターノードをせっていする
MASTER_ADDR=slurm0-a3-ghpc-3
#**********************

MASTER_PORT=65531

NODE_RANK=${1}
echo "Node rank: "$NODE_RANK

NNODES=16
GPUS_PER_NODE=8

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8

model_size=8
NUM_LAYERS=32

HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # hiddensizeの3.5倍
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8

SEQ_LENGTH=2048

# distributed settings
TENSOR_PARALLEL_SIZE=1 
PIPELINE_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=3072
TRAIN_STEPS=300000
LR_DECAY_ITERS=300000

LR=4E-5
MIN_LR=1.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=0.8

# model config
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model"
CHECKPOINT_DIR="/storage5/shared/Llama-3-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-nnodes${NNODES}"
CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-8/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}-nnodes${NNODES}"

log_path="${CHECKPOINT_SAVE_DIR}/log"

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
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")


mkdir -p ${CHECKPOINT_SAVE_DIR}
mkdir -p ${log_path}


# checkpoint load
#if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
#  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
#else
  # first training
#  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"
#fi


#finetune
CHECKPOINT_ARGS="--load /storage5/shared/checkpoints/llama8b_test_tp1_pp4 --finetune"

JOB_NAME="Llama-3-8b-0627_hatakeyama_16nodes"

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
  --init-method-std 0.008 \
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
  --adam-eps 1e-05 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 10000 \
  --eval-iters 10 \
  --bf16 \
  --use-checkpoint-args \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 10000.0 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --attention-softmax-in-fp32 \
  --recompute-activations \
  --recompute-granularity "selective" \
  --transformer-impl "transformer_engine" \
  --fp8-format 'hybrid' \
  --fp8-amax-compute-algo max \
  --fp8-amax-history-len 1024 \
  --no-fp8-wgrad \
  --use-z-loss \
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3-8B" \
  --wandb-entity "weblab-geniac1" \
"

current_time=$(date "+%Y.%m.%d_%H.%M.%S")

log_file="${log_path}/${JOB_NAME}_${node_rank}_${current_time}.log"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


if [[ $node_rank -gt 0 ]]; then
     log_file="${log_path}/${JOB_NAME}_${node_rank}_${current_time}.log"
     run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/shared/jk/Megatron-LM/pretrain_gpt.py
     ${megatron_options} \
     2>&1 | tee ${log_file}"
else
     run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/shared/jk/Megatron-LM/pretrain_gpt.py
     ${megatron_options}"
fi



echo ${run_cmd}
eval ${run_cmd}
set +x

