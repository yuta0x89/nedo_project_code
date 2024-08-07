#!/bin/bash
CONDA_ENV=deep_llama3_v4
SCRIPT_ROOT=/storage5/Megatron-LM/
WANDB_RUN_NAME=llama3
#MASTER_ADDR=hostname
MASTER_PORT=65524
NNODES=2
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
    ssh -q "$node" "source /storage2/miniconda3/etc/profile.d/conda.sh && \
            conda activate ${CONDA_ENV} && \
			export LD_LIBRARY_PATH=/storage2/miniconda3/envs/deep_llama3_v4/lib:$LD_LIBRARY_PATH && \
            cd $SCRIPT_ROOT && \
	        bash /storage5/Megatron-LM/scripts/tsubame/Llama-3-35b/llama-3-35b_2.sh $NODE_RANK" > /storage5/${node}_output.log 2>&1 &
    ((NODE_RANK+=1))
done
wait
exit 0
