#!/bin/bash

#checkpoint conversion
#/storage5/shared/hatakeyama/0611te/Megatron-LM/scripts/tsubame/ckpt/hf_to_megatron_llama3-8b_0627.sh

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#**********************
#マスターノードをせっていする
MASTER_ADDR=slurm0-a3-ghpc-2
#**********************

MASTER_PORT=65533
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

SEQ_LENGTH=8192
#SEQ_LENGTH=2048

# distributed settings
TENSOR_PARALLEL_SIZE=1 
PIPELINE_PARALLEL_SIZE=4
CONTEXT_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1536
TRAIN_STEPS=7400
LR_DECAY_ITERS=7400

LR=0.5E-5
MIN_LR=0.1E-5
LR_WARMUP_STEPS=50
WEIGHT_DECAY=0.1
GRAD_CLIP=0.8

# model config
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model"
CHECKPOINT_DIR="/storage5/shared/Llama-3-8b/0729cleaned_tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-nnodes${NNODES}"
CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-8/0729cleaned_tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}-nnodes${NNODES}"

log_path="${CHECKPOINT_SAVE_DIR}/log"

DATA_PATH_LIST=(
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_6_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_27_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_14_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_26_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_12_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_13_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_15_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_21_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_19_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_30_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_17_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_8_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_22_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_23_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_7_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_10_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_24_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_16_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_20_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_29_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_28_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_9_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_18_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_25_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_11_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b_bacth/logical-wizardlm-7b_b3g_5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b-ja/split_20240728_132437_0_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b-ja/split_20240728_132605_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240729_a/logical-wizardlm-7b-ja/split_20240728_132732_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240728_a/0723out_multiturn_cl_gcp/0_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/kaist-ai_CoT-Collection_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/AtlasUnified_atlas-math-sets_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/jonathanasdf_MathGLM-dataset_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/ChristophSchuhmann_basic-math-problems-with-step-by-step-solutions_b3g_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240727_a/0723math_instructions/math-ai_StackMathQA_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240724_a/WebNovels-Ja_0_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240724_a/WebNovels-Ja_1_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240724_a/WebNovels-Ja_2_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240724_a/WebNovels-Ja_3_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240724_a/WebNovels-Ja_4_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240724_a/WebNovels-Ja_5_text_sentence"
"/storage5/shared/p2_corpus/tokenized_data/20240724_a/WebNovels-Ja_6_text_sentence"

)

# 配列をスペース区切りの文字列に変換
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")


mkdir -p ${CHECKPOINT_SAVE_DIR}
mkdir -p ${log_path}


# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"
fi

#初期化する場合
#CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR} --no-load-rng --no-load-optim"

#finetune
#CHECKPOINT_ARGS="--load /storage5/shared/Llama-3-8/0719cleaned_tp1-pp4-ct1-LR5.0E-5-MINLR0.5E-5-WD0.1-WARMUP500-nnodes16 --finetune"

#loaderとモデル重みのみ読み込んで再開
#CHECKPOINT_ARGS="--load /storage5/shared/Llama-3-8/tp1-pp4-ct1-LR8E-5-MINLR2.0E-5-WD0.1-WARMUP1000-nnodes16 --no-load-optim"

JOB_NAME="Llama-3-8b-0729_hatakeyama_cleaned_synth"

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
  --adam-eps 1e-08 \
  --log-interval 1 \
  --save-interval 300 \
  --eval-interval 10000 \
  --eval-iters 10 \
  --bf16 \
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
  --no-fp8-wgrad \
  --use-z-loss \
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project "Llama-3-8B" \
  --wandb-entity "weblab-geniac1" \
"
#--with-packing \

#--use-checkpoint-args \
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

