#!/bin/bash
#source ~/miniconda3/etc/profile.d/conda.sh
# conda activate share-jk_abeja_py310_TEv1.7_FAv2.5.7
#conda activate oE_py310_TEv1.7_FAv2.5.7

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export NVTE_APPLY_QK_LAYER_SCALING=1

# distributed settings
MASTER_ADDR=slurm0-a3-ghpc-3
MASTER_PORT=65001
NODE_RANK=${1}
echo "Node rank: "$NODE_RANK
#NODE_RANK=0
NNODES=16
GPUS_PER_NODE=8
export NUM_GPU_PER_NODE=8

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

# model config
# mixtral-8x7B-v0.1: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
model_size=8
NUM_LAYERS=32

HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336 # hiddensizeの3.5倍
NUM_HEADS=32
NUM_KEY_VALUE_HEADS=8

SEQ_LENGTH=2048

NUM_EXPERTS=8
NUM_EXPERT_TOP_K=2

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=16
EXPERT_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
#DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=3072
TRAIN_STEPS=125000
LR_DECAY_ITERS=125000

LR=5.0E-5
MIN_LR=5.0E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"

CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0713_3rd_tonyu_tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

DATA_PATH_LIST=(
"/storage5/shared/corpus/phase1_tokenizer_data/tokenized/fineweb/CC-MAIN-2023-50_text_document"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_aq_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ah_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ao_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_aj_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ae_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_an_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_aa_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_al_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_am_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ab_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_af_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ac_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ar_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_as_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ak_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ad_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ap_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ai_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240712_c/twitter_archive_20240712/split_twitter_archive_cleaned_ag_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_c3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_b7_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_e1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_a1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_d2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_b6_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_a6_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_f5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_c4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_f3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_b5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_a3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_f1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_e2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_f2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_b2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_a7_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_f6_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_e4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_a2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_d4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_a5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_b4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_g2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_e5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_a4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_d3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_c2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_b3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_d1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_f4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_d5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_b1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_e6_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_c6_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_c5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_d6_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_e3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_c1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240713_d/llmjp_v4_wo_crean/llmjp_v4_wo_crean_batch_g1_text_sentence"
)


# 配列をスペース区切りの文字列に変換
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
mkdir -p ${CHECKPOINT_SAVE_DIR}

#CHECKPOINT_DIR="/storage5/shared/Llama-3-8-8MoE/hatakeyama_test/0710eight_std0001_chk_0126000"
#CHECKPOINT_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0712_2nd_tonyu_tp1-pp16-ct1-LR5.0E-5-MINLR5.0E-6-WD0.1-WARMUP1000"
CHECKPOINT_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0713_3rd_tonyu_tp1-pp16-ct1-LR5.0E-5-MINLR5.0E-6-WD0.1-WARMUP1000"
#ゼロからのスタート for 速度の最適化
#CHECKPOINT_DIR=$CHECKPOINT_SAVE_DIR

#finetune
# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_DIR} --no-load-rng --no-load-optim"
fi

#再開
#CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"

#特定のcheckpointからの再開 (loader も初期化) 713のチェックポイント切替時のみコメントアウトした。
#ck_restart_dir="/storage5/shared/Llama-3-8b-MoE/8x8_0712_2nd_tonyu_tp1-pp16-ct1-LR5.0E-5-MINLR5.0E-6-WD0.1-WARMUP1000"
#CHECKPOINT_ARGS="--load ${ck_restart_dir} --finetune"


JOB_NAME="Llama-3-8x8b-MoE_0713_3rd_tonyu"

# run
megatron_options="  \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --expert-model-parallel-size ${EXPERT_PARALLEL_SIZE} \
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
  --moe-aux-loss-coeff 0.02 \
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
  --save-interval 800 \
  --eval-interval 10000 \
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
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3-8B-MoE" \
  --wandb-entity "weblab-geniac1" \
  --use-mcore-models \
  --moe-router-load-balancing-type "aux_loss" \
  --transformer-impl "transformer_engine" \
  --log-throughput \
  --no-fp8-wgrad \
  --moe-per-layer-logging \
  --moe-z-loss-coeff 0.00001 \
  --distributed-timeout-minutes 30 \
	"
#  --moe-grouped-gemm \
#  --no-fp8-wgrad \ \\  --use-mcore-models \  --use-checkpoint-args \   --log-throughput \
#
#  --overlap-grad-reduce \
#  --overlap-param-gather \
#  --moe-token-dispatcher-type alltoall \

current_time=$(date "+%Y.%m.%d_%H.%M.%S")

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#run_cmd="torchrun $DISTRIBUTED_ARGS /home/ext_yokobase_gmail_com/ABEJA/Megatron-LM/pretrain_gpt.py
#  ${megatron_options}"
run_cmd="torchrun $DISTRIBUTED_ARGS /storage5/Megatron-LM/pretrain_gpt.py
  ${megatron_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
exit 0
