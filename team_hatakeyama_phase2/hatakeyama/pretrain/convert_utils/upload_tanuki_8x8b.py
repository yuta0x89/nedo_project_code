import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from types import SimpleNamespace

"""
srun --nodelist=slurm0-a3-ghpc-[0] --gpus-per-node=1 --time=30-00:00:00 -c 16 --pty bash -i
srun --nodelist=slurm0-a3-ghpc-[0] --gpus-per-node=0 --time=30-00:00:00 -c 16 --mem=100GB --pty bash -i
source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
#source ~/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
#conda activate llmeval
cd /storage5/shared/hatakeyama/post_training
python upload_tanuki_8x8b.py \
--output_tokenizer_and_model_dir /storage5/someya/outputs/sftlab-experiments/8x8B/someya-sft_007-zero3_multi_node_offload_optimizer/checkpoint-120 \
--huggingface_name 8x8Bsomeya-sft-007-checkpoint-120
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
        trust_remote_code=True,
        )
    return tokenizer, model



def main() -> None:
    args = parse_arguments()

    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(
        args.output_tokenizer_and_model_dir)

    print("model type:",type(local_model))

    local_tokenizer.push_to_hub("team-hatakeyama-phase2/"+args.huggingface_name)
    local_model.push_to_hub("team-hatakeyama-phase2/"+args.huggingface_name)


if __name__ == "__main__":
    main()
