#!/bin/bash

#SBATCH --partition=a3
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --job-name=convert_8x8b_hf
#SBATCH --output=convert_8x8b_hf.out
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=slurm0-a3-ghpc-[7]

#SBATCH -c 48

#SBATCH --mem=600GB

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7

mega_checkpoint_path="/storage5/shared/Llama-3-8/0805cleaned_tp1-pp4-ct1-LR1.0E-5-MINLR1.0E-7-WD0.1-WARMUP100-nnodes16" 
HF_checkpoint_path="/storage5/shared/Llama-3-8/HF/0805cleaned_iter_08100"


cd /storage5/Megatron-LM/tools/checkpoint
python convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir $mega_checkpoint_path \
  --save-dir $HF_checkpoint_path \
  --hf-tokenizer-path /storage5/shared/Llama-3-8/HF/0805cleaned_iter_06000 \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --megatron-path /storage5/shared/jk/Megatron-LM