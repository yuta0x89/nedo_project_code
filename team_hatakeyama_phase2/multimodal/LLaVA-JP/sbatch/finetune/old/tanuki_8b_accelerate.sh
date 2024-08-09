#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-1
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_tanuki_8b_stage2.out
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/finetune/finetune_tanuki_8b_accelerate.sh
