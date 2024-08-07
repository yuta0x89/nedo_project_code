#!/bin/bash
# distributed settings

#sbatch --nodelist=slurm0-a3-ghpc-[6] --gpus-per-node=1 --time=30-00:00:00 --mem=200GB -c 24 /storage5/Megatron-LM/scripts/tsubame/ckpt/megatron_hf_llama3-8b.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oE_py310_TEv1.7_FAv2.5.7

TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=4

ITERATION=49500
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)

# model config
MEGATRON_CHECKPOINT_DIR=/storage5/shared/Llama-3-8/tp1-pp4-ct1-LR8E-5-MINLR2.0E-5-WD0.1-WARMUP1000-nnodes16
HF_CHECKPOINT_DIR=/storage5/shared/Llama-3-8/HF/tp1-pp4-ct1-LR8E-5-MINLR2.0E-5-WD0.1-WARMUP1000-nnodes16/iter_${FORMATTED_ITERATION}

mkdir -p ${HF_CHECKPOINT_DIR}

#echo $ITERATION > "${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR="/storage5/hf_private/model/hatakeyama-llm-team/Tanuki-8B-Instruct" #"/storage5/split/split/split/tokernizer/tokenizer_scale200.model"

# convert
python /storage5/shared/jk/Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --megatron-path /storage5/shared/jk/Megatron-LM
