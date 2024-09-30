#!/bin/bash

# multinode設定
TE_INSTALL_DIR=/storage5/shared/hatakeyama/0611te
SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM

echo script: $SCRIPT_ROOT

#MASTER_PORT=6003
MASTER_PORT=6004
NNODES=2
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
        bash $SCRIPT_ROOT/scripts/tsubame/tanuki-8b/0627_2node_test.sh $NODE_RANK
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done
wait
