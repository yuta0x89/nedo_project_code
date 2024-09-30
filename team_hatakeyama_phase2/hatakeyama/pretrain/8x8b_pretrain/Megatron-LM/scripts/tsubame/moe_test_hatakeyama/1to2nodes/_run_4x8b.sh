#!/bin/bash

#実行コマンドの例
#sbatch --nodelist=slurm0-a3-ghpc-[6] --gpus-per-node=8 --time=30-00:00:00 --mem=1000GB -c 100 /storage5/Megatron-LM/scripts/tsubame/tanuki-8x8B/tanuki-8x8b-tp4.sh
#sbatch --nodelist=slurm0-a3-ghpc-[0,1] --gpus-per-node=2 --time=30-00:00:00 --mem=200GB -c 64 /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/_run_multi.sh
#sbatch --nodelist=slurm0-a3-ghpc-[15] --gpus-per-node=8 --time=30-00:00:00 --mem=1000GB -c 108 /storage5/Megatron-LM/scripts/tsubame/tanuki-8x8B/_run_tanuki.sh

export TE_INSTALL_DIR=/storage5
SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM
MASTER_PORT=65001
NNODES=1
chmod +x $SCRIPT_ROOT/scripts/tsubame/moe_test_hatakeyama/tanuki-4x8b.sh
#mapfile -t NODES < <(scontrol show hostname)
NODES=(
    "slurm0-a3-ghpc-1"
    "slurm0-a3-ghpc-2"
)
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    ssh -q $node "cd $SCRIPT_ROOT && \
        bash $SCRIPT_ROOT/scripts/tsubame/moe_test_hatakeyama/tanuki-4x8b.sh $NODE_RANK
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done
wait
