#!/bin/bash

#実行コマンドの例
#sbatch --nodelist=slurm0-a3-ghpc-[6] --gpus-per-node=8 --time=30-00:00:00 --mem=1000GB -c 100 /storage5/Megatron-LM/scripts/tsubame/tanuki-8x8B/tanuki-8x8b-tp4.sh
#sbatch --nodelist=slurm0-a3-ghpc-[0,1] --gpus-per-node=2 --time=30-00:00:00 --mem=200GB -c 64 /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/_run_multi.sh
#sbatch --nodelist=slurm0-a3-ghpc-[15] --gpus-per-node=8 --time=30-00:00:00 --mem=1000GB -c 108 /storage5/Megatron-LM/scripts/tsubame/tanuki-8x8B/_run_tanuki.sh


TE_INSTALL_DIR=/storage5
SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM
MASTER_PORT=65003
NNODES=16
chmod +x /storage5/shared/hatakeyama/0706moe_abeja/script/tanuki-4x8b-tp4.sh
#mapfile -t NODES < <(scontrol show hostname)
NODES=(
    "slurm0-a3-ghpc-4"
    "slurm0-a3-ghpc-5"
    "slurm0-a3-ghpc-7"
    "slurm0-a3-ghpc-8"
    "slurm0-a3-ghpc-9"
    "slurm0-a3-ghpc-10"
    "slurm0-a3-ghpc-11"
    "slurm0-a3-ghpc-12"
    "slurm0-a3-ghpc-13"
    "slurm0-a3-ghpc-14"
    "slurm0-a3-ghpc-15"
    "slurm0-a3-ghpc-16"
    "slurm0-a3-ghpc-17"
    "slurm0-a3-ghpc-18"
    "slurm0-a3-ghpc-19"
    "slurm0-a3-ghpc-20"
)
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    ssh -q $node "source /storage5/shared/abeja/miniconda3/etc/profile.d/conda.sh && \
        conda activate share-jk_abeja_py310_TEv1.7_FAv2.5.7 && \
        cd $SCRIPT_ROOT && \
        bash /storage5/shared/hatakeyama/0706moe_abeja/script/tanuki-4x8b-tp4.sh $NODE_RANK" > /storage5/Megatron-LM/scripts/tsubame/tanuki-8x8B/log/${node}_output_${current_time=}.log 2>&1 &
    ((NODE_RANK+=1))
done
wait
