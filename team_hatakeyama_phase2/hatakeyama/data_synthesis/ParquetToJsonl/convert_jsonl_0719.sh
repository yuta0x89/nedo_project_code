#!/bin/sh
#SBATCH --partition=a3
#SBATCH --time=30-00:00:00
#SBATCH --gpus-per-node=0
#SBATCH --nodelist=slurm0-a3-ghpc-[6]
#SBATCH -c 4
#SBATCH --mem=100GB

export PATH=/home/ext_kan_hatakeyama_s_gmail_com/miniconda3/envs/textprocess/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate textprocess
echo "begin conversion"
################
#command例
#cd /storage5/shared/corpus/synthetic/ParquetToJsonl/
#sbatch --nodelist=slurm0-a3-ghpc-[0] --gpus-per-node=0 --mem=200GB --time=30-00:00:00 -c 8 convert_jsonl_0701_gcp_cc.sh 

# cc from hf cleaned
python parquet_to_jsonl_ca.py --input_path /storage5/shared/corpus/synthetic/SyntheticTexts/0716ca_auto_instruct --out_dir ../../2nd_tonyu/hf_synth/synth_text_gcp_ca0719 --column_name text --do_clean false