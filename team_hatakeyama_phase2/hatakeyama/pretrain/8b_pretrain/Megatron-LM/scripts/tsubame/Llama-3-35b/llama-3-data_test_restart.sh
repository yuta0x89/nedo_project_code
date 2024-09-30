#!/bin/bash

#export LD_LIBRARY_PATH="$CONDA_ENV/envs/.te/lib:$LD_LIBRARY_PATH"
#export PATH="$CONDA_ENV/envs/.te/bin:$PATH"
#conda activate .te

#private 環境
#source ~/miniconda3/etc/profile.d/conda.sh  
#conda activate .te
#export LD_LIBRARY_PATH="$CONDA_ENV/envs/.te/lib"
#export PATH="$CONDA_ENV/envs/.te/bin:$PATH"


#共用環境
source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

#**********************
#マスターノードをせっていする
MASTER_ADDR=slurm0-a3-ghpc-0
#**********************

MASTER_PORT=65532
#NODE_RANK=${1}
echo "Node rank: "$NODE_RANK
NODE_RANK=0

NNODES=1

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=1

# model config
# llama-3-8b: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
#HIDDEN_SIZE=4096
#FFN_HIDDEN_SIZE=14336 # intermediate size (HuggingFace)
#NUM_LAYERS=32
#NUM_HEADS=32
#NUM_KEY_VALUE_HEADS=8

model_size=7
NUM_LAYERS=8

HIDDEN_SIZE=256
FFN_HIDDEN_SIZE=1024 # hiddensizeの3.5倍
#ffn_hidden_size: 24576 # hiddensizeの4倍
#num_attn_heads: 48
NUM_HEADS=4
NUM_KEY_VALUE_HEADS=4

SEQ_LENGTH=2048

# distributed settings
TENSOR_PARALLEL_SIZE=1  # fixed
PIPELINE_PARALLEL_SIZE=1 # num layers 32: Llama-2 8B
CONTEXT_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=4
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=1.0E-5
MIN_LR=1.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="/storage5/tokenizer/ja53K_en13K_add_dummy.model"
CHECKPOINT_DIR="/storage5/shared/Llama-3-1b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}"
CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-1b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

log_path="${CHECKPOINT_SAVE_DIR}/log"

#TRAIN_DATA_PATH="/storage5/shared/k_nishizawa/tokenized_20240610/tokenized_data/_text_document"

DATA_PATH_LIST=(
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch3_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch5_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch14_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch11_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch9_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch6_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch15_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch4_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch13_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch8_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch1_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch7_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch10_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch12_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch2_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/llmjp/batch0_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/PMC/merged_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2020-05_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2017-22_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2020-29_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2022-40_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2023-23_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2021-39_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2014-41_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2019-22_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2017-47_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2018-39_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2017-17_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2019-39_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2021-21_text_sentence"
"/storage5/shared/corpus/2nd_tonyu/tokenized/fineweb-edu/CC-MAIN-2017-13_text_sentence"
)
# 配列をスペース区切りの文字列に変換
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")

echo $TRAIN_DATA_PATH

mkdir -p ${CHECKPOINT_SAVE_DIR}
mkdir -p ${log_path}

MASTER_PORT=65532

# checkpoint load
#if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
#  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
#else
#  # first training
#  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"
#fi

#ここでオリジナルのチェックポイントを指定する
load_checkpoint_dir=/storage5/shared/Llama-3-1b/tp1-pp1-ct1-LR1.0E-5-MINLR1.0E-6-WD0.1-WARMUP1000
CHECKPOINT_ARGS="--load ${load_checkpoint_dir} --finetune"

JOB_NAME="Llama-3-7b-0620hatakeyama_data_test_2epoch"

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
  --rope-theta 500000.0 \
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
  --use-z-loss \
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3-1B" \
  --wandb-entity "weblab-geniac1" \
	"

current_time=$(date "+%Y.%m.%d_%H.%M.%S")

log_file="${log_path}/llma3_31_${node_rank}_${current_time}.log"

DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPU_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"



echo "dist args"
echo $DISTRIBUTED_ARGS
echo ""

#共用のmegatronを使う

#run_cmd="torchrun $DISTRIBUTED_ARGS $TE_INSTALL_DIR/Megatron-LM/pretrain_gpt.py
run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/Megatron-LM/pretrain_gpt.py
 ${megatron_options} \
 2>&1 | tee ${log_file}"

echo ${run_cmd}
eval ${run_cmd}
set +x