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
MASTER_ADDR=slurm0-a3-ghpc-2
MASTER_PORT=65002
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

#SEQ_LENGTH=2048
SEQ_LENGTH=4096
#SEQ_LENGTH=8192

NUM_EXPERTS=8
NUM_EXPERT_TOP_K=2

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=16
EXPERT_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
#DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=3072
TRAIN_STEPS=125000
LR_DECAY_ITERS=125000

LR=5.0E-6
MIN_LR=1.0E-6
LR_WARMUP_STEPS=50
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"

CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0801_8th_tonyu_tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

DATA_PATH_LIST=(
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/0731automatic_math_hatakeyama_batch/0731automatic_math_hatakeyama_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/0731automatic_math_hatakeyama_batch/0731automatic_math_hatakeyama_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/0731automatic_math_hatakeyama_batch/0731automatic_math_hatakeyama_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/0731automatic_math_hatakeyama_batch/0731automatic_math_hatakeyama_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/0731automatic_math_hatakeyama_batch/0731automatic_math_hatakeyama_b3g_5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/arithmetic_qa_by_rules_20240801/outputs.arithmetic.1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/arithmetic_qa_by_rules_20240801/outputs.arithmetic.2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/arithmetic_qa_by_rules_20240801/outputs.arithmetic.3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/arithmetic_qa_by_rules_20240801/outputs.arithmetic.4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/arithmetic_qa_by_rules_20240801/outputs.arithmetic.5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/tosho_batch/tosho_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/tosho_batch/tosho_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/tosho_batch/tosho_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240801_a/tosho_batch/tosho_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/AtlasUnified_atlas-math-sets_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/kaist-ai_CoT-Collection_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/no_pack_20240731_b/0723math_instructions/math-ai_StackMathQA_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/inst_merged_0727_v2/inst_merged_0727_v2/inst_merged_0727_v2_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/inst_merged_0727_v2/inst_merged_0727_v2/inst_merged_0727_v2_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/inst_merged_0727_v2/inst_merged_0727_v2/inst_merged_0727_v2_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/inst_merged_0727_v2/inst_merged_0727_v2/inst_merged_0727_v2_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/inst_merged_0727_v2/inst_merged_0727_v2/inst_merged_0727_v2_b3g_5_text_sentence"
)

#退避
#"/storage5/shared/p2_corpus/4096_packed_data/4096_tokenized_data/20240730_b/hanrei/2020_packing_text_sentence"
#"/storage5/shared/p2_corpus/4096_packed_data/4096_tokenized_data/20240730_a/logicaltext-wizardlm8x22b-api/split_20240725_082543_0_packing_text_sentence"
#"/storage5/shared/p2_corpus/4096_packed_data/4096_tokenized_data/starcoder_2/starcoder/starcoder_b5g11_packing_text_sentence"

# 配列をスペース区切りの文字列に変換
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
mkdir -p ${CHECKPOINT_SAVE_DIR}

CHECKPOINT_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0801_8th_tonyu_tp1-pp16-ct1-LR5.0E-5-MINLR5.0E-6-WD0.1-WARMUP500"

#ゼロからのスタート for 速度の最適化
#CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0730_saitekika_tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"
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



#特定のcheckpointからの再開 (loader も初期化) チェックポイント切替時のみコメントアウト
ck_restart_dir="/storage5/shared/Llama-3-8b-MoE/8x8_0730_7th_tonyu_tp1-pp16-ct1-LR5.0E-6-MINLR4.0E-6-WD0.1-WARMUP50"
CHECKPOINT_ARGS="--load ${ck_restart_dir} --finetune"


JOB_NAME="Llama-3-8x8b-MoE_0801_8th_tonyu"

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
  --dataloader-type cyclic \
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
  --save-interval 400 \
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
  --rope-theta 500000.0"

#  --with-packing \
#  --skip-train-iteration-range 200-300 \
 
#--skip-train-iteration-range 2500-2700 8814-9100 \
  #best fit
#--rope-theta 500000.0 \
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
