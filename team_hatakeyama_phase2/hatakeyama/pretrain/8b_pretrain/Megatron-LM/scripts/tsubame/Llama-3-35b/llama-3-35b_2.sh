#!/bin/bash
#export LD_LIBRARY_PATH="$CONDA_ENV/envs/.te/lib:$LD_LIBRARY_PATH"
#export PATH="$CONDA_ENV/envs/.te/bin:$PATH"

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
#

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NODE_RANK=${1}
echo "Node rank: "$NODE_RANK


# distributed settings
#**********************
#マスターノードをせっていする
MASTER_ADDR=slurm0-a3-ghpc-19
#**********************

#MASTER_PORT=65532
MASTER_PORT=65533

# NODE_RANK=0

NNODES=2
GPUS_PER_NODE=8
chmod +x llama-3-35b_2.sh
echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8

# model config
# llama-3-8b: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
#HIDDEN_SIZE=4096
#FFN_HIDDEN_SIZE=14336 # intermediate size (HuggingFace)
#NUM_LAYERS=32
#NUM_HEADS=32
#NUM_KEY_VALUE_HEADS=8

model_size=39
NUM_LAYERS=72

HIDDEN_SIZE=6912
FFN_HIDDEN_SIZE=16128 # hiddensizeの4.5倍にすること
#ffn_hidden_size: 24576 # hiddensizeの4倍にしたパターン
#num_attn_heads: 48
NUM_HEADS=54
NUM_KEY_VALUE_HEADS=9

SEQ_LENGTH=2048

# distributed settings
TENSOR_PARALLEL_SIZE=1  # fixed
PIPELINE_PARALLEL_SIZE=16 # num layers 32: Llama-2 8B
CONTEXT_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1536
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=1.0E-4
MIN_LR=0.99E-4
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="/storage5/tokenizer/ja53K_en13K_add_dummy.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"
CHECKPOINT_DIR="/storage5/shared/Llama-3-35b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}"
CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-35/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

log_path="${CHECKPOINT_SAVE_DIR}/log"

TRAIN_DATA_PATH="/storage5/shared/k_nishizawa/tokenized_20240610/tokenized_data/_text_document"

mkdir -p ${CHECKPOINT_SAVE_DIR}


# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"
fi

JOB_NAME="Llama-3-35b-2nodes"

# run

#--train-tokens ${TRAIN_TOKENS}
megatron_options="  \
  --distributed-timeout-minutes 30 \
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
	--tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --train-iters ${TRAIN_STEPS} \
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
  --save-interval 10 \
  --eval-interval 100 \
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
  --wandb-project "Llama-3-35B" \
  --wandb-entity "weblab-geniac1" \
	"

current_time=$(date "+%Y.%m.%d_%H.%M.%S")

echo "current time:"$current_time

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [[ $NODE_RANK -gt 0 ]]; then
    log_file="${log_path}/llma3_35B_${NODE_RANK}_${current_time}.log"
    run_cmd="torchrun $DISTRIBUTED_ARGS $TE_INSTALL_DIR/Megatron-LM/pretrain_gpt.py
     ${megatron_options} \
     2>&1 | tee ${log_file}"
else
    run_cmd="torchrun $DISTRIBUTED_ARGS $TE_INSTALL_DIR/Megatron-LM/pretrain_gpt.py
     ${megatron_options}"
fi
echo "command for node rank: "$NODE_RANK
echo "--------"
echo ${run_cmd}
echo "--------"

eval ${run_cmd}
set +x
exit 0
