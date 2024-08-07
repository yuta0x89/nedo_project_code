#!/bin/bash

#実行コマンドの例
#sbatch --nodelist=slurm0-a3-ghpc-[3-18] --gpus-per-node=8 --time=30-00:00:00 --mem=1000GB -c 128 _run_multi_16_batch.sh

export TE_INSTALL_DIR=/storage5/shared/jk
SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM

echo script: $SCRIPT_ROOT

WANDB_RUN_NAME=llama3
MASTER_PORT=6004
chmod +x 0627_2node_test.sh
#mapfile -t NODES < <(scontrol show hostname)
NODES=(
    "slurm0-a3-ghpc-6"
    "slurm0-a3-ghpc-15"
)


NODE_RANK=0
for node in "${NODES[@]}"; do
    devices=$(ssh -q $node "echo $CUDA_VISIBLE_DEVICES")
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    
    echo "SSH command sent for node: $node with node rank of $NODE_RANK"
    echo ""
    
    ssh -q $node "
        cd $SCRIPT_ROOT && \
        bash /storage5/shared/hatakeyama/0611te/Megatron-LM/scripts/tsubame/tanuki-8b/0627_2node_test.sh $NODE_RANK
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done
wait
