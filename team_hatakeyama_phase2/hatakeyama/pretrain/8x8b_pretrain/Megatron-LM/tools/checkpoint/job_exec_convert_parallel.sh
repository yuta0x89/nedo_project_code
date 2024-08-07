#!/bin/bash

#SBATCH --partition=a3
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --job-name=convert__prallel_8x8b
#SBATCH --output=convert_parallel_8x8b.out
#SBATCH --gpus-per-node=3
#SBATCH --nodelist=slurm0-a3-ghpc-[6]

#SBATCH -c 72

#SBATCH --mem=800GB

export CUDA_DEVICE_MAX_CONNECTIONS=1
source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7

mega_checkpoint_path="/storage5/shared/Llama-3-8b-MoE/8x8_0724_6th_tonyu_tp1-pp16-ct1-LR2.5E-5-MINLR5.0E-6-WD0.1-WARMUP500" 
HF_checkpoint_path="/storage5/shared/Nishijima/Llama-3-8b-MoE/6th_tonyu_iter_0012800"
mega_save_path="/storage5/shared/Nishijima/Llama-3-8b-MoE/mega/6th_tonyu_iter_0012800"
TARGET_TP=2
TARGET_PP=8

cd /storage5/Megatron-LM/tools/checkpoint
python convert.py \
  --model-type GPT \
  --loader tanuki_moe_mcore \
  --saver tanuki_moe_hf \
  --load-dir $mega_checkpoint_path \
  --save-dir $HF_checkpoint_path \
  --hf-tokenizer-path /storage5/shared/Nishijima/Llama-3-8b-MoE/5th_tonyu_iter_20000 \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --megatron-path /storage5/shared/jk/Megatron-LM 

python convert.py \
  --model-type GPT \
  --loader tanuki_moe_hf \
  --saver tanuki_mcore \
  --load-dir $HF_checkpoint_path \
  --save-dir $mega_save_path \
  --tokenizer-model /storage5/shared/Nishijima/Llama-3-8b-MoE/iter_12800 \
  --target-tensor-parallel-size $TARGET_TP \
  --target-pipeline-parallel-size $TARGET_PP