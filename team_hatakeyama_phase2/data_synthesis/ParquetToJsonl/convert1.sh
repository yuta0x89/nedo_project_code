#!/bin/sh
#SBATCH --partition=a3
#SBATCH --time=30-00:00:00
#SBATCH --gpus-per-node=0
#SBATCH --nodelist=slurm0-a3-ghpc-[0]
#SBATCH -c 4
#SBATCH --mem=100GB


export PATH=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/textprocess/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate textprocess

################
#command例
#cd /storage5/shared/corpus/synthetic/ParquetToJsonl/
#sbatch --nodelist=slurm0-a3-ghpc-[20] --gpus-per-node=0 --mem=200GB --time=30-00:00:00 -c 8 convert.sh 

#sansu
#python parquet_to_jsonl.py --input_path ../from_huggingface/Sansu/ --out_dir ../../2nd_tonyu/hf_synth/sansu --do_clean false --column_name answer

#synthetic text
python parquet_to_jsonl.py --input_path ../from_huggingface/SyntheticText/data --out_dir ../../2nd_tonyu/hf_synth/synthtext

#math
python parquet_to_jsonl.py --input_path ../from_huggingface/SyntheticTextOpenMathInstruct/data --out_dir ../../2nd_tonyu/hf_synth/openmath

#wiki
python parquet_to_jsonl.py --input_path ../from_huggingface/SyntheticTextWikiTranslate/data --out_dir ../../2nd_tonyu/hf_synth/wiki --column_name ja

#cc uncleaned
#python parquet_to_jsonl.py --input_path /storage5/shared/corpus/synthetic/SyntheticTexts/out_data --out_dir ../../2nd_tonyu/hf_synth/0619synth_gcp_needed_clean --column_name ja

#cc from hf
#python parquet_to_jsonl.py --input_path /storage5/shared/corpus/synthetic/from_huggingface/SyntheticTextCCUncleaned/data --out_dir ../../2nd_tonyu/hf_synth/0619synth_hf_needed_clean --column_name ja