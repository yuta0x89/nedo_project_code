import sys
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, MixtralConfig

def main():

    import argparse
    parser = argparse.ArgumentParser(
        description="Megatron Checkpoint Utility Arguments",
        allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--loader-org', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--loader-con', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    
    known_args, _ = parser.parse_known_args()
    args = parser.parse_args()
    args.weight_check=True

    model = AutoModelForCausalLM.from_pretrained(args.loader_org)
    state_dict = model.state_dict()
    
    #del model

    ref_model = AutoModelForCausalLM.from_pretrained(args.loader_con)
    ref_state_dict = ref_model.state_dict()

    #del ref_model

    for key in ref_state_dict:
        if key not in state_dict:
            print(f'Key {key} not found in state_dict')
    assert sorted(list(ref_state_dict.keys())) == sorted(list(state_dict.keys()))
    for key in state_dict:
        print(f"Checking {key}")
        # shapeは確実に一致することを確認. embeddingのweightはvocab_sizeがおなじの時だけチェック
        if (ref_model.vocab_size != model.vocab_size) and ("embed_tokens" in key or "lm_head" in key):
            print(f"Skip shape check for {key}")
        else:
            assert ref_state_dict[key].shape == state_dict[key].shape, f"Shape of {key} not equal"
        # かなり小さい小数点の違いで一致しないので、np.iscloseでチェック
        # 以下のようなtorch.equalは一致しない
        # assert torch.equal(ref_state_dict[key], state_dict[key]), f"Key {key} not equal"
        if args.weight_check:
            if (ref_model.vocab_size != model.vocab_size) and ("embed_tokens" in key or "lm_head" in key):
                    print(f"Skip weight check for {key}")
            else:
                check_atol_list = [1e-4, 1e-5]  # 時間かかるので2個だけ
                ref_numpy = ref_state_dict[key].cpu().numpy()
                converted_numpy = state_dict[key].cpu().numpy()
                for atol in check_atol_list:
                    if not np.all(np.isclose(ref_numpy, converted_numpy, atol=atol)):
                        # 1e-5でチェックすればよほどOKなはず
                        if atol >= 1e-5:
                            raise AssertionError(f"Key {key} not equal with atol {atol}")
                        else:
                            print(f"Key {key} not equal with atol {atol}")
    print("Check passed.")

if __name__ == "__main__":
    main()