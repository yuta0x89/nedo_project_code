#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
cd /bb/llm/gaf51275/llama/Megatron-LM
source .env/bin/activate

DATASET_DIR=/bb/llm/gaf51275/llm-jp/datasets/llm-jp-corpus/108b-merged
OUTPUT_DIR=/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/tokenized/llama-2-hf-tokenizer

mkdir -p ${OUTPUT_DIR}

# tokenize japanese wikipedia
python tools/preprocess_data.py \
  --input ${DATASET_DIR}/en_wiki.jsonl \
  --output-prefix ${OUTPUT_DIR}/en_wiki \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model /bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-hf/tokenizer.model \
  --append-eod \
  --workers 64
