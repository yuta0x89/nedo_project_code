#!/bin/bash

#実行コマンドの例
#sbatch --nodelist=slurm0-a3-ghpc-[0,19] --gpus-per-node=8 --time=30-00:00:00 --mem=500GB -c 64 _run_multi_2.sh



SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM/
WANDB_RUN_NAME=llama3
#MASTER_PORT=6003
MASTER_PORT=6006
NNODES=2

#実行権限をつける
chmod +x llama-3-35b_2.sh


#mapfile -t NODES < <(scontrol show hostname)
NODES=("slurm0-a3-ghpc-19" "slurm0-a3-ghpc-20")
echo "SCRIPT_ROOT: "$SCRIPT_ROOT

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
        bash $SCRIPT_ROOT/scripts/tsubame/Llama-3-35b/llama-3-35b_2.sh $NODE_RANK
    " 2>&1 | while IFS= read -r line; do
        echo "[$node] $line"
    done &

    ((NODE_RANK+=1))
done
wait
#exit 0
