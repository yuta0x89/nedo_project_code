import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from types import SimpleNamespace

"""
srun --nodelist=slurm0-a3-ghpc-[0] --gpus-per-node=1 --time=30-00:00:00 -c 16 --pty bash -i
source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
#source ~/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
#conda activate llmeval
cd /storage5/shared/hatakeyama/post_training
python upload_tanuki.py \
--output_tokenizer_and_model_dir /storage5/personal/shioya/po_model/polab-experiments/8B/pass4_exp002-dpo_full_006-zero2 \
--huggingface_name 0719cleaned_tp1-pp4-ct1-hf_iter_0006400-to-pass4_exp001-old_template_07-zero1-to-pass4_exp002-dpo_full_006-zero2
"""
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_tokenizer_and_model_dir",
                        type=str, required=True)
    parser.add_argument("--huggingface_name", type=str, required=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    #tokenizerはいつも同じ

    tokenizer_name="/storage5/shared/Llama-3-8/HF/cont_0126000_lr1_5e_m4"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        input_tokenizer_and_model_dir, device_map="auto",
        torch_dtype=torch.bfloat16,
        )
    return tokenizer, model



def main() -> None:
    args = parse_arguments()

    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(
        args.output_tokenizer_and_model_dir)


    local_tokenizer.push_to_hub("team-hatakeyama-phase2/8b-iter-"+args.huggingface_name)
    local_model.push_to_hub("team-hatakeyama-phase2/8b-iter-"+args.huggingface_name)


if __name__ == "__main__":
    main()
