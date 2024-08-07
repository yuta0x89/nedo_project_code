#!/bin/bash

#SBATCH --partition=a3
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --job-name=convert_8x8b_MG
#SBATCH --output=convert_8x8b_MG.out
#SBATCH --gpus-per-node=3
#SBATCH --nodelist=slurm0-a3-ghpc-[17]

#SBATCH -c 72

#SBATCH --mem=800GB

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7

HF_checkpoint_path="/storage5/shared/Nishijima/Llama-3-8b-MoE/6th_tonyu_iter_0012800"
mega_save_path="/storage5/shared/Nishijima/Llama-3-8b-MoE/6th_tonyu_iter_0012800"
TARGET_TP=2
TARGET_PP=8

python convert.py \
  --model-type GPT \
  --loader tanuki_moe_hf \
  --saver tanuki_mcore \
  --load-dir $HF_checkpoint_path \
  --save-dir $mega_save_path \
  --tokenizer-model /storage5/shared/Nishijima/Llama-3-8b-MoE/iter_12800 \
  --target-tensor-parallel-size $TARGET_TP \
  --target-pipeline-parallel-size $TARGET_PP