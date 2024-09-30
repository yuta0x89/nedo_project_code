#!/bin/bash

#SBATCH --partition=a3
#SBATCH --time=30-00:00:00
#SBATCH --nodes=2
#SBATCH --job-name=llama35-2node-test
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[0-1]

#SBATCH -c 60
#SBATCH --mem=500GB

cd /storage5/shared/hatakeyama/0611te/Megatron-LM/scripts/tsubame/Llama-3-35b
bash _run_multi_2.sh