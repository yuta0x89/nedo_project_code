#!/bin/bash

#実行コマンドの例
#sbatch --nodelist=slurm0-a3-ghpc-[3-18] --gpus-per-node=8 --time=30-00:00:00 --mem=1000GB -c 128 _run_multi_16_batch.sh

#export TE_INSTALL_DIR=/strorage5/shared/jk
SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM

echo script: $SCRIPT_ROOT

WANDB_RUN_NAME=llama3
#MASTER_ADDR=hostname
MASTER_PORT=6003
#MASTER_PORT=6004
NNODES=16
chmod +x llama-3-35b_16_pp4_tp4_2nd_dataset_rerestart_0626.sh
#mapfile -t NODES < <(scontrol show hostname)
NODES=(
    "slurm0-a3-ghpc-3"
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
    "slurm0-a3-ghpc-16"
    "slurm0-a3-ghpc-17"
    "slurm0-a3-ghpc-18"
    "slurm0-a3-ghpc-19"
    "slurm0-a3-ghpc-20"
)


NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    ssh -q $node "
        conda activate .te && \
		export LD_LIBRARY_PATH=$CONDA_ENV/envs/.te/lib:$LD_LIBRARY_PATH && \
        cd $SCRIPT_ROOT && \
        bash $SCRIPT_ROOT/scripts/tsubame/Llama-3-35b/llama-3-35b_16_pp4_tp4_2nd_dataset_rerestart_0626.sh $NODE_RANK
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done
wait
