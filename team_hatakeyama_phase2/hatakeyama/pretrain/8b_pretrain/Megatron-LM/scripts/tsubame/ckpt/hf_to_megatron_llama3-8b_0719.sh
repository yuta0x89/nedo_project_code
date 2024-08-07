#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=1:00:00
#$ -o outputs/convert/hf_megatron/$JOB_ID
#$ -e outputs/convert/hf_megatron/$JOB_ID
#$ -p -5

# Load modules
#module use ~/modulefiles

#module load ylab/cuda/12.1
#module load ylab/cudnn/8.9.7
#module load ylab/nccl/cuda-12.1/2.18.3
#module load ylab/hpcx/2.17.1
#module load ninja/1.11.1

# swich virtual env
#source .env/bin/activate
source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=4

# model config
HF_CHECKPOINT_DIR=/storage5/shared/Llama-3-8/cleaned_tp1-pp2-ct1-LR5E-5-MINLR0.5E-5-WD0.1-WARMUP1000-nnodes1/hf_iter_0003200
MEGATRON_CHECKPOINT_DIR=/storage5/shared/checkpoints/8b-0719-add-iter3200_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/storage5/llm/models/hf/0524snapshot/tokenizer.json

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
