#!/bin/bash
#iterationを指定する
find /var/tmp -maxdepth 1 -user ext_kan_hatakeyama_s_gmail_com -print0 | xargs -0 rm -rf

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
#


MEGATRON_CHECKPOINT_DIR=$1
ITERATION=$2
hf_name=$3
FORMATTED_ITERATION=$(printf "%07d" $ITERATION)
HF_CHECKPOINT_DIR=${MEGATRON_CHECKPOINT_DIR}/hf_iter_${FORMATTED_ITERATION}

echo "megatron data: $MEGATRON_CHECKPOINT_DIR"
echo "iter: $ITERATION"
echo "hf_dir: $HF_CHECKPOINT_DIR"


# model config
mkdir -p ${HF_CHECKPOINT_DIR}
echo $ITERATION >"${MEGATRON_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"

# tokenizer config
TOKENIZER_MODEL_DIR=/storage5/shared/corpus/phase1_tokenizer_data/tokernizer
#cd /storage5/shared/hatakeyama/0611te/Megatron-LM
cd /storage5/Megatron-LM
# convert
#if [ ! -d "$HF_CHECKPOINT_DIR" ]; then
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama3_hf \
  --load-dir ${MEGATRON_CHECKPOINT_DIR} \
  --save-dir ${HF_CHECKPOINT_DIR} \
  --hf-tokenizer-path ${TOKENIZER_MODEL_DIR} \
  --save-dtype bfloat16 \
  --loader-transformer-impl transformer_engine \
  --megatron-path /storage5/Megatron-LM
#else
#  echo "HF_CHECKPOINT_DIR exists."
#fi

  #--megatron-path /storage5/shared/hatakeyama/0611te/Megatron-LM
echo "convert done!"
cd /storage5/shared/hatakeyama/post_training
python upload_tanuki.py --output_tokenizer_and_model_dir $HF_CHECKPOINT_DIR --huggingface_name $hf_name

echo "upload done!"