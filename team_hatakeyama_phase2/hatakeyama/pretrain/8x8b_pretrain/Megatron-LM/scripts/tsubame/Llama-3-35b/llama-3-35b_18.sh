#!/bin/bash
export LD_LIBRARY_PATH=/storage2/miniconda3/envs/deep_llama3_v4/lib:$LD_LIBRARY_PATH

#source /storage2/miniconda3/etc/profile.d/conda.sh
conda activate deep_llama3_v4

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# distributed settings
MASTER_ADDR=slurm0-a3-ghpc-3
MASTER_PORT=65532
NODE_RANK=${1}
echo "Node rank: "$NODE_RANK
# NODE_RANK=0

NNODES=16
GPUS_PER_NODE=8

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

# model config 
# llama3-70B
#HIDDEN_SIZE=8192
#FFN_HIDDEN_SIZE=28672 # intermediate size (HuggingFace)
#NUM_LAYERS=80
#NUM_HEADS=64
#NUM_KEY_VALUE_HEADS=8

# model config 
# qwen 1.5 32B
#HIDDEN_SIZE=5120
#FFN_HIDDEN_SIZE=27392 # intermediate size (HuggingFace)
#NUM_LAYERS=64
#NUM_HEADS=40
#NUM_KEY_VALUE_HEADS=8

# model config 
# Yi 35B
HIDDEN_SIZE=7168
#FFN_HIDDEN_SIZE=20480 # original
FFN_HIDDEN_SIZE=25088 # x 3.5
#NUM_LAYERS=60
NUM_LAYERS=58
NUM_HEADS=56
NUM_KEY_VALUE_HEADS=8


##hatakeyama llama3 35B
#model_size=35
#NUM_LAYERS=54
#HIDDEN_SIZE=6912
#FFN_HIDDEN_SIZE=24192 # hiddensizeの4倍にしたパターン
#NUM_HEADS=54
#NUM_KEY_VALUE_HEADS=9

##hatakeyama llama3 35B_v2
#model_size=35
#NUM_LAYERS=54
#HIDDEN_SIZE=6912
#FFN_HIDDEN_SIZE=24192 # hiddensizeの4倍にしたパターン
#NUM_HEADS=54
#NUM_KEY_VALUE_HEADS=9

SEQ_LENGTH=2048

#code llama 35b
#HIDDEN_SIZE=8192
#FFN_HIDDEN_SIZE=22016 # intermediate size (HuggingFace)
#NUM_LAYERS=48
#NUM_HEADS=64
#NUM_KEY_VALUE_HEADS=8

# distributed settings
TENSOR_PARALLEL_SIZE=1  # fixed
PIPELINE_PARALLEL_SIZE=32 # num layers 32: Llama-2 8B
CONTEXT_PARALLEL_SIZE=1
# 24:15:47 start
# training config
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=2304 #1512 #2304 #1536
TRAIN_STEPS=12500
LR_DECAY_ITERS=12500

LR=1.0E-5
MIN_LR=1.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="/storage5/tokenizer/ja53K_en13K_add_dummy.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"
CHECKPOINT_DIR="/storage5/Llama-3-35b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}"
CHECKPOINT_SAVE_DIR="/storage5/Llama-3-35/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

log_path="${CHECKPOINT_SAVE_DIR}/log"

TRAIN_DATA_PATH="/storage5/shared/k_nishizawa/tokenized_20240610/tokenized_data/_text_document"

mkdir -p ${CHECKPOINT_SAVE_DIR}
mkdir -p ${log_path}

MASTER_PORT=65532

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"
fi

JOB_NAME="Llama-3-35b-exp2"

# run

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
  --save-interval 1000 \
  --eval-interval 1000 \
  --eval-iters 1000 \
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
  --wandb-entity "weblab-geniac5" \
	"



current_time=$(date "+%Y.%m.%d_%H.%M.%S")

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#if [[ $node_rank -gt 0 ]]; then
#    log_file="${log_path}/llma3_35B_${node_rank}_${current_time}.log"
#    run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/Megatron-LM/pretrain_gpt.py
#     ${megatron_options} \
#     2>&1 | tee ${log_file}"
#else
run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/Megatron-LM/pretrain_gpt.py
  ${megatron_options}"
#fi

echo ${run_cmd}
eval ${run_cmd}
set +x
exit 0
