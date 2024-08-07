#!/bin/bash
CONDA_ENV=deep_llama3_v4
SCRIPT_ROOT=/storage5/Megatron-LM/
WANDB_RUN_NAME=llama3
#MASTER_ADDR=hostname
MASTER_PORT=65532
NNODES=1
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
    ssh -q $node "conda activate ${CONDA_ENV} && \
			export LD_LIBRARY_PATH=/storage2/miniconda3/envs/deep_llama3_v4/LD_LIBRARY_PATH && \
            cd $SCRIPT_ROOT && \
	    bash /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/llama-3-35b_18.sh \
	    $NODE_RANK" &

    ((NODE_RANK+=1))
done
exit 0
