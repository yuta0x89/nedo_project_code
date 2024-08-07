#!/bin/bash

echo "START RUN"

source ~/miniconda3/etc/profile.d/conda.sh
# conda activate share-jk_abeja_py310_TEv1.7_FAv2.5.7
conda activate oE_py310_TEv1.7_FAv2.5.7

NNODES=1
WORLD_SIZE=8
TMP_SIZE=2
PMP_SIZE=2
LOAD_CHECKPOINT_PATH=/storage5/shared/hatakeyama/0706moe_abeja/mergekit/megatron_core/model_two
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8

MASTER_ADDR=slurm0-a3-ghpc-1
MASTER_PORT=65003
NODE_RANK=${1}

TRAIN_DATA_PATH="/storage5/shared/corpus/phase1_tokenizer_data/tokenized/synth/0619synth_gcp_needed_clean_text_document"
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"


GPT_ARGS="
    --tensor-model-parallel-size $TMP_SIZE \
    --pipeline-model-parallel-size $PMP_SIZE \
    --expert-model-parallel-size 1 \
    --context-parallel-size 1 \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 100 \
    --bf16 \
    --swiglu \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --no-position-embedding \
    --num-experts 2 \
    --position-embedding-type rope \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --no-masked-softmax-fusion \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --no-load-optim \
    --no-load-rng \
    --no-masked-softmax-fusion \
    --skip-train \
    --use-mcore-models \
    --moe-router-load-balancing-type "aux_loss" \
    --transformer-impl "transformer_engine" \
    --log-throughput \
    --no-fp8-wgrad \
    --moe-per-layer-logging \
"

DATA_ARGS="
    --data-path $TRAIN_DATA_PATH \
    --split 949,50,1
"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS /storage5/Megatron-LM/foward_test_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    --distributed-backend nccl \
    --load $LOAD_CHECKPOINT_PATH