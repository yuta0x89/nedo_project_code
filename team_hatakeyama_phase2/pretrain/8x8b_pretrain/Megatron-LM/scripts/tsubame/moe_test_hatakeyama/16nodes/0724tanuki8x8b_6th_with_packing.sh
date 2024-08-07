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
MASTER_PORT=65003
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

LR=2.5E-5
MIN_LR=5.0E-6
LR_WARMUP_STEPS=500
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="/storage5/split/split/tokernizer/tokenizer_scale200.model" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"

CHECKPOINT_SAVE_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0724_6th_tonyu_tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

DATA_PATH_LIST=(
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_9_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_4_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_6_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_15_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_19_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_10_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_14_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_7_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_17_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_5_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_11_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_12_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_3_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_2_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_1_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_18_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_8_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_16_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/en_open_webmath_batch/en_open_webmath_batch_batch5g_13_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/ja_law_200m/ja_law_batch_200m_3_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/ja_law_200m/ja_law_batch_200m_1_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/20240724_a/ja_law_200m/ja_law_batch_200m_2_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g13_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g8_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g14_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g6_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g7_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g9_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g19_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g16_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g5_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g20_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g17_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g10_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g4_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g2_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g11_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g18_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g12_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g3_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g15_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/fineweb-edu/2022_a/2022_a/fineweb_edu_2022_b5g1_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g3_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g4_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g9_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g27_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g13_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g10_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g23_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g25_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g14_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g22_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g5_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g29_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g2_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g18_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g17_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g33_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g34_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g32_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g11_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g8_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g26_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g16_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g20_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g31_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g15_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g7_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g21_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g24_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g28_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g12_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g19_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g6_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g30_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/0615llmjp_corpus_v2/0615llmjp_corpus_v2/llmjp_corpus_v2_b5g1_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a1_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a4_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a3_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a2_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a10_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a5_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a6_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a8_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a9_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/llmjp_corpus_v5_2019_2020_a/llmjp_corpus_v5_2019_2020_a/0713llmjp_corpus_v5_2019_2020_batch_a7_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g11_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g8_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g1_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g5_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g4_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g10_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g12_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g6_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g3_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g9_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g2_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/starcoder/starcoder/starcoder_b5g7_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_12_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_112_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_110_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_116_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_115_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_13_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_18_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_16_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_19_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_118_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_15_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_111_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_14_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_114_packing_text_sentence"
"/storage5/shared/p2_corpus/packed_data/tokenized_data/cosmopedia-v2/cosmopedia-v2_1/integ_b2g_17_packing_text_sentence"
)



# 配列をスペース区切りの文字列に変換
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
mkdir -p ${CHECKPOINT_SAVE_DIR}

#CHECKPOINT_DIR="/storage5/shared/Llama-3-8-8MoE/hatakeyama_test/0710eight_std0001_chk_0126000"
#CHECKPOINT_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0712_2nd_tonyu_tp1-pp16-ct1-LR5.0E-5-MINLR5.0E-6-WD0.1-WARMUP1000"
CHECKPOINT_DIR="/storage5/shared/Llama-3-8b-MoE/8x8_0724_6th_tonyu_tp1-pp16-ct1-LR5.0E-5-MINLR5.0E-6-WD0.1-WARMUP1000"
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

#特定のcheckpointからの再開 (loader も初期化) チェックポイント切替時のみコメントアウト
#iterは20000に指定した。それ以降は2epochなので。
#ck_restart_dir="/storage5/shared/Llama-3-8b-MoE/8x8_0720_5th_tonyu_tp1-pp16-ct1-LR5.0E-5-MINLR5.0E-6-WD0.1-WARMUP500"
#CHECKPOINT_ARGS="--load ${ck_restart_dir} --finetune"


JOB_NAME="Llama-3-8x8b-MoE_0724_6th_tonyu"

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
  --with-packing \
  --skip-train-iteration-range 2500-2700 8814-9100 \
	"
  #best fit
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
