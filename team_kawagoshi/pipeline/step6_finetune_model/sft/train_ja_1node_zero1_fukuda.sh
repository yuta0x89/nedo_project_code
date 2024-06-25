#!/bin/bash

set -e
echo ""

# Change the below configurations here
ucllm_nedo_dev_train_dir="/storage2/ucllm_nedo_prod/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"

echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
# Initializes the arguments.
input_model_name_or_path="/storage2/output/finetune/ja_edinet_ft_math_epoch_10/checkpoint/SFT_zero1_2024.05.19_13.35.40/checkpoint-60"
OUTPUT_DIR="/storage2/output/SFT/math_test"
wandb_entity="weblab-geniac5"
wandb_project="SFT"

# Prints the arguments.
echo "input_model_name_or_path = ${input_model_name_or_path}"
echo "output_dir = ${OUTPUT_DIR}"
echo "wandb_entity = ${wandb_entity}"
echo "wandb_project = ${wandb_project}"
echo "${OUTPUT_DIR##*/}"
echo ""

mkdir -p ${OUTPUT_DIR}

current_time=$(date "+%Y.%m.%d_%H.%M.%S")
jobname="SFT_zero1_${current_time}"

username=$(whoami)
log_path="${OUTPUT_DIR}/log"
CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint/${jobname}"
# CHECKPOINT_PATH="${OUTPUT_DIR}/fukuda/checkpoint/${jobname}"

mkdir -p ${log_path}
mkdir -p ${CHECKPOINT_PATH}

host="${HOSTNAME}"

wandb_options=" \
    --use_wandb \
    --wandb_entity ${wandb_entity} \
    --wandb_project ${wandb_project} \
    --wandb_group finetune_${OUTPUT_DIR##*/}_${host}_${current_time}"

# Finetunes the pretrained model.

deepspeed ${ucllm_nedo_dev_train_dir}/llm-jp-sft/train_ja_kawagoshi.py \
    --num_train_epochs 85 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 2e-3 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --bf16 \
    --data_files OpenMathInstruct-1-1.8m-ja-120000-converted-nosp.jsonl \
    --sample_sizes 320 \
    --model_name_or_path ${input_model_name_or_path} \
    --output_dir ${CHECKPOINT_PATH} \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_step 20 \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --load_best_model_at_end \
    --deepspeed /storage2/ucllm_nedo_prod/train/scripts/step4_finetune_model/pre_dev/deepspeed_config/ds_config_zero2_nooffload.json \
    --save_only_model \
    --use_flash_attention_2 \
    ${wandb_options} \
    2>&1 | tee ${log_path}/${host}_${current_time}.log

echo ""
echo "Finished to finetune the pretrained model."
echo ""

# --sample_sizes 1 1 1 1 1 1 1 1000 \
# oasst2-33k-ja_converted-nosp.jsonl oasst1-21k-ja_converted-nosp.jsonl OpenMathInstruct-1-1.8m-ja-12000-converted-nosp.jsonl amenokaku-code-instruct-merged.jsonl HachiML-Evol-hh-rlhf-gen3-1k-merged.jsonl HachiML-Hachi-Alpaca-merged.jsonl kunishou-databricks-dolly-15k-ja-instructions-merged.jsonl AnswerCarefullyVersion001_Dev-merged.jsonl oasst2-33k-en_converted-nosp.jsonl

#, oasst1-21k-en-merged oasst2-33k-ja-merged oasst2-33k-en-merged 