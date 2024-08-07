#!/bin/bash

#export PATH=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/textprocess/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate synthtext
export HF_HOME=/storage5/shared/huggingface_cache2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
python 0715_ca_auto_instruct.py $SLURM_JOB_ID
