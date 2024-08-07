#!/bin/bash

#SBATCH --partition=a3
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --job-name=tp_test_8x8b
#SBATCH --output=tp_test.out
#SBATCH --gpus-per-node=8
#SBATCH --nodelist=slurm0-a3-ghpc-[6,17]

#SBATCH -c 72

#SBATCH --mem=800GB

export TE_INSTALL_DIR=/storage5
SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM
NNODES=2
#mapfile -t NODES < <(scontrol show hostname)
NODES=(
    "slurm0-a3-ghpc-6"
    "slurm0-a3-ghpc-17"
)
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    ssh -q $node "cd $SCRIPT_ROOT && \
        bash /storage5/Megatron-LM/scripts/tsubame/tanuki-8x8B/tanuki-8x8b-tp16.sh $NODE_RANK
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done
wait
