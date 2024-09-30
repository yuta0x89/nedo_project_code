#!/bin/bash
#iterationを指定する
find /var/tmp -maxdepth 1 -user ext_kan_hatakeyama_s_gmail_com -print0 | xargs -0 rm -rf

source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
#

cd /storage5/shared/hatakeyama/post_training
python upload_tanuki_38b.py --output_tokenizer_and_model_dir /storage5/shared/Llama-3-35/HF/Llama-3-35b-16nodes_2nd_tonyu-tp4-pp4-ct1-LR2E-5-MINLR1.99E-5-WD0.1-WARMUP8000/iter_0006000 --huggingface_name 38b-2nd-tonyu-iter6000


echo "upload done!"