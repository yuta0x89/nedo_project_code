#!/bin/bash

SCRIPT_ROOT=$TE_INSTALL_DIR/Megatron-LM
WANDB_RUN_NAME=llama3
#MASTER_ADDR=hostname
MASTER_PORT=6003
#MASTER_PORT=6004
NNODES=20
#NNODES=2
mapfile -t NODES < <(scontrol show hostname)
NODE_RANK=0
for node in "${NODES[@]}"
do
    echo "node: "$node
    echo $NODE_RANK
    devices=`ssh -q $node "echo $CUDA_VISIBLE_DEVICES"`
    gpu_count=$(echo $devices | tr ',' '\n' | wc -l)
    echo "gpu_count: "$gpu_count
    #ssh -q $node "source $HOME/.pyenv/versions/miniconda3/etc/profile.d/conda.sh && \
    ssh -q $node "conda activate .te && \
			export LD_LIBRARY_PATH=$CONDA_ENV/envs/.te/lib:$LD_LIBRARY_PATH && \
            cd $SCRIPT_ROOT && \
	    bash $SCRIPT_ROOT/scripts/tsubame/Llama-3-35b/llama-3-35b_20.sh \
	    $NODE_RANK" &

    ((NODE_RANK+=1))
done
exit 0
