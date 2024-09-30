#!/bin/bash

# Command line options go here
#SBATCH --time=96:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-1
#SBATCH --job-name=llava-jp
#SBATCH --output=sbatch_logs/train_llava-jp_stage2.out
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500GB

# Command(s) goes here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava-jp

#cd /storage5/multimodal/LLaVA-JP
cd /storage5/multimodal/work/yamaguchi/LLaVA-JP

bash scripts/finetune/finetune_accelerate.sh \
    ./configs/train/finetune/batch_128_mlp6x_gelu.json \
    ./configs/image_encoder/siglip-so400m-patch14-384.json \
    ./configs/dataset/v1_stage_2_curation.json \
    ./configs/model/tanuki-8b.json \
    ./output_llava/checkpoints/finetune-llava-jp-Tanuki-8B-vision-v1_curation_mlp6x_gelu \
    llava-jp-stage2 \
    Tanuki-8B-vision-v1_curation_mlp6x_gelu \
    ./output_llava/checkpoints/pretrain-llava-jp-Tanuki-8B-vision-v1_mlp6x_gelu/mm_projector.bin
