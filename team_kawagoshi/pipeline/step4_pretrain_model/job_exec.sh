#!/bin/bash

#SBATCH --partition=a3
#SBATCH --time=30-00:00:00
#SBATCH --nodes=3
#SBATCH --job-name=kawagoshi_llm_run
#SBATCH --output=kawagoshi_llm_run.out
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[3-5]

#SBATCH -c 100

#SBATCH --mem=1000GB

# 新しい内容を ~/.bashrc に書き込む
cat > ~/.bashrc << 'EOF'
# Conda configuration
source /storage2/miniconda3/etc/profile.d/conda.sh
conda activate .venv

# Set file descriptor limit
ulimit -n 262144

# Environment variables
export PRETRAIN="/storage2/nedo_project_code/team_kawagoshi/pipeline/step4_pretrain_model"
export EXP_HOME="/storage2"
EOF

echo "新しい設定が ~/.bashrc に適用されました。"
source ~/.bashrc

echo "~/.bashrc を実行しました。"
# 新しい内容を ~/.ssh/config に書き込む
bash /storage2/nedo_project_code/team_kawagoshi/pipeline/common/create_ssh_config_file_for_gcp_play_multi_node_multi_gpu.sh
echo "新しい設定が ~/.ssh/config に適用されました。"

function chmode_at_exit(){
    chmod -R 777 /storage2/ucllm_nedo_prod;
    chmod -R 777 /storage2/output;
    chmod -R 777 /storage2/wandb;
    chmod -R 777 /storage2/datasets;
    chmod -R 777 /storage2/tokenizer;
    chmod -R 777 /storage2/kawagoshi_llm_run.out;
}

trap chmode_at_exit EXIT

bash $PRETRAIN/pretrain_llama3_3node.sh
