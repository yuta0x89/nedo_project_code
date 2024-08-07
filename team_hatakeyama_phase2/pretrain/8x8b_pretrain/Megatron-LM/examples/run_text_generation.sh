#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
# conda activate share-jk_abeja_py310_TEv1.7_FAv2.5.7
conda activate oE_py310_TEv1.7_FAv2.5.7

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export NVTE_APPLY_QK_LAYER_SCALING=1

# distributed settings
MASTER_ADDR=slurm0-a3-ghpc-2
MASTER_PORT=65012
NODE_RANK=${1}
echo "Node rank: "$NODE_RANK
#NODE_RANK=0
NNODES=1
GPUS_PER_NODE=4

NUM_LAYERS=8

HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # hiddensizeの3.5倍
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8

NUM_EXPERTS=8

TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=1
EXPERT_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1

SEQ_LENGTH=2048

echo "MASTER_ADDR=${MASTER_ADDR}"

#CHECKPOINT=<Path to checkpoint (e.g /345m)>
#VOCAB_FILE=<Path to vocab.json (e.g. /gpt2-vocab.json)>
#MERGE_FILE=<Path to merges.txt (e.g. /gpt2-merges.txt)>
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"
#CHECKPOINT_DIR="/storage5/shared/Nishijima/mergekit/megatron_model/model_four_tp4_pp1"
#CHECKPOINT_DIR="/storage5/shared/Nishijima/mergekit/model_core/random_8_layer_tp2 "
CHECKPOINT_DIR="/storage5/shared/Nishijima/mergekit/model_core/random_merge_8_layer_tp2"
CHECKPOINT_DIR="/storage5/shared/Nishijima/Llama-3-8b-MoE/megatron/3rd_tonyu_iter_800_8layers"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

megatron_options="  \
       --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
       --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
       --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
       --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE} \
       --use-distributed-optimizer \
       --load ${CHECKPOINT_DIR}  \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEADS} \
       --num-query-groups ${NUM_KEY_VALUE_HEADS} \
       --num-experts ${NUM_EXPERTS} \
       --tokenizer-type SentencePieceTokenizer \
       --group-query-attention \
       --tokenizer-model ${TOKENIZER_MODEL} \
       --max-position-embeddings ${SEQ_LENGTH} \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length ${SEQ_LENGTH}  \
       --seed 42 \
       --distributed-backend nccl \
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
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
       --use-mcore-models \
       --moe-router-load-balancing-type "aux_loss" \
       --transformer-impl "transformer_engine" 
       --no-fp8-wgrad \
      "

run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/Megatron-LM/tools/run_text_generation.py \
  ${megatron_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
exit 0