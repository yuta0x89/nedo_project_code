#!/bin/bash

#実行コマンドの例
#sbatch --nodelist=slurm0-a3-ghpc-[3-18] --gpus-per-node=8 --time=30-00:00:00 --mem=1000GB -c 128 /storage5/shared/Nishijima/_run_multi_16_pp4_tp4_batch_shared.sh
#sbatch --nodelist=slurm0-a3-ghpc-[19,20] --gpus-per-node=2 --time=30-00:00:00 --mem=200GB -c 32 /storage5/shared/Nishijima/_run_wandb_test.sh
#sbatch --nodelist=slurm0-a3-ghpc-[1] --gpus-per-node=2 --time=30-00:00:00 --mem=200GB -c 32 /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/_run_multi.sh
#sbatch --nodelist=slurm0-a3-ghpc-[0,1] --gpus-per-node=2 --time=30-00:00:00 --mem=200GB -c 64 /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/_run_multi.sh


TE_INSTALL_DIR=/storage5
SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM
WANDB_RUN_NAME=llama3
#MASTER_ADDR=hostname
MASTER_PORT=65001
#MASTER_PORT=6004
NNODES=1
chmod +x /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/_run_multi.sh
#mapfile -t NODES < <(scontrol show hostname)
NODES=(
    "slurm0-a3-ghpc-1"
)

NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    ssh -q $node "source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh && \
        conda activate share-jk_py310_TEv1.7_FAv2.5.7 && \
        cd $SCRIPT_ROOT && \
        bash /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/llama-3-35b_1.sh $NODE_RANK" > /storage5/${node}_output.log 2>&1 &
    ((NODE_RANK+=1))
done
wait
