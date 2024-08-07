import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from types import SimpleNamespace


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
    tokenizer_name="/storage5/shared/Llama-3-35/HF/Llama-3-35b-16nodes_2nd_tonyu-tp4-pp4-ct1-LR2E-5-MINLR1.99E-5-WD0.1-WARMUP8000/iter_0006000"
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


    local_tokenizer.push_to_hub("team-hatakeyama-phase2/38b-iter-"+args.huggingface_name)
    local_model.push_to_hub("team-hatakeyama-phase2/38b-iter-"+args.huggingface_name)


if __name__ == "__main__":
    main()
