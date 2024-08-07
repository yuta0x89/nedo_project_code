#!/bin/bash

#export path=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/textprocess/bin:$path
source ~/miniconda3/etc/profile.d/conda.sh && conda activate synthtext
export hf_home=/storage5/shared/huggingface_cache2
export ld_library_path=$conda_prefix/lib:$ld_library_path
export path=$conda_prefix/bin:$path
python 0721free_multiturn_with_memory.py $slurm_job_id
