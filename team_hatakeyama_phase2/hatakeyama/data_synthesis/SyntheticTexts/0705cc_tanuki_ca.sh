#!/bin/bash

#中身は0715だが､shel scriptの停止用に､script名は同じにしておく

#export PATH=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/textprocess/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate synthtext

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
export HF_HOME=/storage5/shared/huggingface_cache2

#python 0721free_multiturn_with_memory.py  $SLURM_JOB_ID
#logical
#python 0802logical_multiturn.py  $SLURM_JOB_ID

#マルチターン
python 0801fixed_multiturn.py  $SLURM_JOB_ID

#8x22b
#python 0805wz_8x22b.py  $SLURM_JOB_ID
