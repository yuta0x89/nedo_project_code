#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/convert/hf_megatron/$JOB_ID
#$ -e outputs/convert/hf_megatron/$JOB_ID
#$ -p -5

# swich virtual env
#source .env/bin/activate
source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2

# model config
HF_CHECKPOINT_DIR=/storage5/shared/Llama-3-8/HF/cont_0126000_lr1_5e_m4
MEGATRON_CHECKPOINT_DIR=/storage5/shared/Llama-3-8/iter126000_pp2_pp1

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/storage5/shared/Llama-3-8/HF/cont_0126000_lr1_5e_m4

cd /storage5/shared/jk/Megatron-LM
# convert
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama3_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16 \
  --saver-transformer-impl "transformer_engine"
