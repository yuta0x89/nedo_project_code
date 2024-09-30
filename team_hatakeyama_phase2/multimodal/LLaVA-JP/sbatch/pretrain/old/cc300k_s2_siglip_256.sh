#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-19
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage1.out
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=200GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/pretrain/pretrain_accelerate_s2_siglip_256.sh configs/train/pretrain/cc300k_s2_siglip_256.json
