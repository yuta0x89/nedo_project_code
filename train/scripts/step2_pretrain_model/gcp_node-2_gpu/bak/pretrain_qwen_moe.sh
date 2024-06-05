#!/bin/bash

set -e
echo ""

# CONDAの.venvの中で定義されている
# GENIAC_ROOT_DIR=/persistentshare/storage/team_haijima

# Stores the directory paths as variables.
root_dir="${GENIAC_ROOT_DIR}/GENIAC_haijima/train"
megatron_deepspeed_dir="${root_dir}/Megatron-DeepSpeed"
echo "root_dir = ${root_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
# TODO: 正式なSetencePieceモデルに置き換える。
input_tokenizer_file="${GENIAC_ROOT_DIR}/datasets_pre/20240418_preprocess/tokenizer/merged.model"
output_model_dir="${GENIAC_ROOT_DIR}/train_results/pretrained_models"
save_interval=1000
wandb_entity="weblab-geniac7"
wandb_project="pretrain_qwen_moe"

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --output_model_dir) output_model_dir=${2}; shift ;;
        --save_interval) save_interval=${2}; shift ;;
        --wandb_entity) wandb_entity=${2}; shift ;;
        --wandb_project) wandb_project=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]] || [[ -z ${output_model_dir} ]] || [[ -z ${wandb_entity} ]] || [[ -z ${wandb_project} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file> --output_model_dir <output_model_dir> --wandb_entity <wandb_entity> --wandb_project <wandb_project>"
    exit 1
fi

# Modifies the arguments.
output_model_dir="${output_model_dir%/}"  # Removes a trailing slash "/" if it exists.

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "output_model_dir = ${output_model_dir}"
echo "save_interval = ${save_interval}"
echo "wandb_entity = ${wandb_entity}"
echo "wandb_project = ${wandb_project}"
echo ""

###############################################################################
### Main configs
## The "GPT-3 XXX" below are configs from GPT-3 paper
## https://arxiv.org/abs/2005.14165, choose based on
## your desired model size or build your own configs

## init_std is standard deviation for weight initialization. Usually larger
## model needs lower std. We used a heuristic equation of sqrt(1/3/hidden_size)
## from the MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)

## We changed min_lr to a lower number (1.0e-6), which we found is able to
## provide better zero-shot eval results.

## Qwen@ MoE 14B --active params 2.7B
# configはhfを参照：https://huggingface.co/docs/transformers/main/model_doc/qwen2_moe#transformers.Qwen2MoeConfig.moe_intermediate_size
seq_len=2048 #3072  # 2048  # 2048 or 3072
#model="qwen_14B_hf_moe"
model="qwen_14B_hf_moe_half"
model_size=14
num_layers=24 #36  # 24 # 36
hidden_size=1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #1536 #2048 #3072 #1536 2048
ffn_hidden_size=4224 #6144 #   4224  # 6144 #5632
moe_hidden_size=1056 #1536 #1056 # 1536 #1408
num_attn_heads=24 #24 #24 # 12 # 24 16
num_key_value_heads=6 # 6 #6 #6 #6 #6 #6 #6 #4 # 3 # 4 Qwen2 uses GQA.
#window_size=4096 # if -1, local attention won't be applied
window_size=2048 # if -1, local attention won't be applied
paged_kv_block_size=0 # if 0, paged atten won't be used
moe_type="hf_mixtral" #HFのMoE
#num_experts=60
num_experts=30 # 20とかで試してみる？
#topk=4
topk=2
normalization="rmsnorm"
norm_eps=1e-6
global_batch_size=1536
max_position_embeddings=32768 # following HF implementaiton, mistral suppports 4096*32 at maximum
moe_loss_coef=0.01
#FIXME
lr=1.0e-4
min_lr=1.0e-6
init_std=0.008

# Deepspeed MoE configs
# DeepspeedのMoEのみ関係
ep_parallel_size=8 # set num_gpus if num_experts > num_gputs else num_experts
moe_train_cap_factor=1.0
moe_eval_cap_factor=1.0
moe_min_cap=4
moe_drop_token="true"


###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens.
train_tokens_in_billion=300
train_tokens=$((${train_tokens_in_billion} * 1000 * 1000 * 1000))

## train_samples is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
train_samples=$(( 300 * 1000 * 1000 * 1000 * 2 / ${seq_len} ))

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B since when batch size warmup is not
## used, there are more tokens per step. Thus we need to increase warmup tokens
## to make sure there are enough warmup steps, which is important for training
## stability.
lr_warmup_tokens_in_million=3000
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000 * 1000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000 * 1000 * 1000))
lr_decay_style="cosine"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=3 #6 #6 #1 #4 #2 # 4 #2 #6 #1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=1 #6

# If you plan to use Megatron-DeepSpeed's deepspeed_to_transformers.py to convert
# the checkpoint from Megatron-DeepSpeed format to Hugging Face Transformers format,
# then sets no_pp to false (even if pp_size is 1).
# The reason why is because Megatron-DeepSpeed's deepspeed_to_transformers.py assumes
# there are "layer_*.pt" files, and "layer_*.pt" files are created if no_pp is false.
# In other words, if no_pp is true, then "layer_*.pt" files are not created and
# Megatron-DeepSpeed's deepspeed_to_transformers.py would fail.
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=2

## Total number of GPUs.
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node="${NHOSTS}"
NHOSTS=3
num_gpus=$((${num_gpus_pernode} * ${num_node}))

## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
#batch_size=$(( ${global_batch_size} / ${dp_size} ))
batch_size=3
###############################################################################
### Misc configs
log_interval=10
eval_iters=10
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
activation_checkpoint="true"
#activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234
num_workers=0

# TODO: フォルダ構成を調整する。また、スクリプトとしても分離したい。
# If either arxiv_text_document.bin or arxiv_text_document.idx doesn't exist yet,
# then downloads arxiv.jsonl and preprocesses the data.
#data_path="${megatron_deepspeed_dir}/dataset/arxiv_text_document"
data_path="/storage3/tokenized_data/merged_120B_text_document"
if [ ! -f "${data_path}.bin" ] || [ ! -f "${data_path}.idx" ]; then
    echo "Either ${data_path}.bin or ${data_path}.idx doesn't exist yet, so download arxiv.jsonl and preprocess the data."
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl \
        --directory-prefix ${megatron_deepspeed_dir}/dataset/
    mv ${megatron_deepspeed_dir}/dataset/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl ${megatron_deepspeed_dir}/dataset/arxiv.jsonl
    python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model ${input_tokenizer_file} \
        --input ${megatron_deepspeed_dir}/dataset/arxiv.jsonl \
        --output-prefix ${megatron_deepspeed_dir}/dataset/arxiv \
        --dataset-impl mmap \
        --workers $(grep -c ^processor /proc/cpuinfo) \
        --append-eod
else
    echo "Both ${data_path}.bin and ${data_path}.idx already exist."
fi
echo ""

prescale_grad="true"
jobname="${model}_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}"
if [[ $zero_stage -gt 0 ]]; then
    jobname="${jobname}_z${zero_stage}"
    prescale_grad="false"
fi
if [[ $mp_size -gt 1 ]]; then
    jobname="${jobname}_mp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
    jobname="${jobname}_pp${pp_size}"
fi
jobname="${jobname}_seed${seed}_rebase"

username=$(whoami)
log_path="${output_model_dir}/log"
checkpoint_path="${output_model_dir}/checkpoint/${jobname}"
tensorboard_path="${output_model_dir}/tensorboard/${jobname}_${host}_${current_time}"
deepspeed_config_dir="${output_model_dir}/deepspeed_config"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
mkdir -p ${deepspeed_config_dir}
###############################################################################
data_options=" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --data-path ${data_path} \
    --data-impl mmap"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --ffn-hidden-size ${ffn_hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --num-key-value-heads ${num_key_value_heads} \
    --seq-length ${seq_len} \
    --window-size ${window_size} \
    --paged-kv-block-size ${paged_kv_block_size} \
    --swiglu \
    --use-rotary-position-embeddings \
    --rotary-percent 0.25 \
    --normalization ${normalization} \
    --layernorm-epsilon ${norm_eps} \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --disable-bias-linear \
    --no-query-key-layer-scaling \
    --rotary-percent 0.25 \
    --max-position-embeddings ${max_position_embeddings} \
    --moe-expert-parallel-size ${ep_parallel_size} \
    --num-experts ${num_experts} \
    --topk ${topk} \
    --moe-type ${moe_type} \
    --moe-hidden-size ${moe_hidden_size} \
    --moe-aux-loss \
    --moe-loss-coeff ${moe_loss_coef} \
    --moe-train-capacity-factor ${moe_train_cap_factor} \
    --moe-eval-capacity-factor ${moe_eval_cap_factor} \
    --moe-min-capacity ${moe_min_cap} \
    --train-tokens ${train_tokens} \
    --train-samples ${train_samples} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --fp16 \
    --seed ${seed} \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn-v2 \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path}"

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

config_json="${deepspeed_config_dir}/ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}.json"
template_json="${megatron_deepspeed_dir}/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
      > ${config_json}

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --no-pipeline-parallel"
#    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

# W&B options.
# Note: the W&B options use "underscore" (like `--use_wandb`) instead of "hyphen" (like `--use-wandb`).
# Note: the W&B group name `--wandb_group` should be less than 128 characters.
wandb_options=" \
    --use_wandb \
    --wandb_entity ${wandb_entity} \
    --wandb_project ${wandb_project} \
    --wandb_group pretrain_gpt_${model_size}B_${host}_${current_time}"

# Creates a hostfile.
script_dir=$(dirname "$0")
hostfile="${script_dir}/hostfile_jobid-${SLURM_JOB_ID}"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)

for node in $nodes
do
  gpu_count=$(ssh ${node} "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
  echo "${node} slots=${gpu_count}"
done > "${hostfile}"

echo "hostfile = ${hostfile}"
cat ${hostfile}
echo ""

deepspeed --hostfile ${hostfile} --master_port 42432 \
    ${megatron_deepspeed_dir}/pretrain_mistral.py \
    ${megatron_options} \
    ${data_options} \
    ${deepspeed_options} \
    ${wandb_options} \
