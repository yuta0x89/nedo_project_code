import json
import os
import sys
import torch
import transformers
from tqdm import tqdm
import types
from pretrain_gpt import model_provider
from transformers import LlamaForCausalLM, MixtralForCausalLM
import torch
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed
from megatron.training.arguments import (parse_args, validate_args)
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import set_global_variables, get_args
from megatron.core.enums import ModelType
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.legacy import fused_kernels
from megatron.core import mpu

def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-expert-model-parallel-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )
    parser.add_argument(
        "--mg-ckpt-path",
        type=str
    )

    return parser


def load_megatron_model(ckpt):
    model = model_provider()

    model_path = ckp
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)
    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.tensor_model_parallel_size
    if args.num_experts is not None:
        num_local_experts = args.num_experts // args.expert_model_parallel_size
    state_dict = {}
    mid_state = defaultdict(list)


    num_layers = args.num_layers // args.pipeline_model_parallel_size
    layers_to_copy = {}
    for tp_rank in range(args.tensor_model_parallel_size):
        for ep_rank in range(args.expert_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                layer_offset = pp_rank * num_layers
                for layer in range(num_layers):
                    pp_layer_id = layer + layer_offset
                    layers_to_copy[f"decoder.layers.{layer}"] = pp_layer_id

                if args.expert_model_parallel_size > 1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, True,
                                                            ep_rank)
                elif args.expert_model_parallel_size == 1:
                    checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank,
                                                            False)
                print(f'load {checkpoint_name}')
                split_state = torch.load(checkpoint_name, map_location="cpu")['model']
                for k, v in split_state.items():
                    if 'local_experts' in k and 'norm' not in k:
                        local_expert_rank = name_to_expert_rank(k)
                        expert_rank = local_expert_rank + num_local_experts * ep_rank
                        k = k.replace(f'local_experts.{local_expert_rank}', f'local_experts.{expert_rank}')
                        mid_state[k].append(v)
                    elif ep_rank == 0:
                        mid_state[k].append(v)
                    try:
                        pattern = re.compile(r'\d+')
                        res = pattern.findall(k)
                        k = re.sub(r"decoder.layers.\d+", "decoder.layers." + str(layers_to_copy["decoder.layers." + res[0]]), k)
                        mid_state[k].append(v)
                    except:
                        mid_state[k].append(v)
    for k, v in mid_state.items():
        if not isinstance(v[0], torch.Tensor) or 'norm' in k or 'router' in k or 'gate' in k:
            target_v = v[0]
        elif 'embedding' in k or 'output_layer' in k:
            target_v = torch.cat(v, dim=0)
        elif 'linear_proj' in k or 'linear_fc2' in k:
            target_v = torch.cat(v, dim=1)
        elif 'linear_qkv.weight' in k:
            viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
            target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
        elif 'linear_qkv.bias' in k:
            viewed = [x.view(group_per_split, -1) for x in v]
            target_v = torch.cat(viewed, dim=0).view(-1)
        elif 'linear_fc1' in k:
            viewed = [x.view(2, -1, args.hidden_size) for x in v]
            target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
        else:
            raise ValueError
        state_dict[k] = target_v

    model.load_state_dict(state_dict, strict=True)
    return model

def check_hf_mg_forward(hfmodel, mgmodel, mgargs):
    hf_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    hidden_size = mgargs.hidden_size
    vocab_size = mgargs.padded_vocab_size


    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hf':
            hf_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == 'mg' and 'layer' in mode:
            mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
        elif frame == 'mg':
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
        if mode in ['hf-lmhead']:
            hf_hiddens[layer_idx][name] = output.transpose(0, 1).reshape(-1, vocab_size)
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
            hf_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
        elif mode in ['mg-lmhead']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, vocab_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
            mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
        elif mode in ['hf-o_proj_out']:
            hf_hiddens[layer_idx][name] = output
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-o_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hf-attn_out']:
            hf_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['mg-attn_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)

    hfmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hf-lmhead'),
                                          with_kwargs=True)

    mgmodel.output_layer.register_forward_hook(
        partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'), with_kwargs=True)

    for idx, layer in enumerate(hfmodel.model.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-layer_in'), with_kwargs=True)

        layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hf-o_proj_in'),
                                                         with_kwargs=True)

        layer.self_attn.o_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-o_proj_out'),
                                                     with_kwargs=True)

        layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hf-attn_out'),
                                              with_kwargs=True)


    for idx, layer in enumerate(mgmodel.decoder.layers):

        layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)

        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True)

        layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'),
                                                   with_kwargs=True)


    input_ids = torch.tensor([[14,   8506,  22564,  27608,  75188,   4344, 1215,  6191,  39554, 36689]]).long().cuda()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    print(hfmodel)
    print(mgmodel)
    is_oom = False
    with torch.inference_mode():
        try:
            hfmodel.cuda()
            hflogits = hfmodel(input_ids=input_ids).logits
        except torch.cuda.OutOfMemoryError:
            print('oom for huggingface model forward')
            is_oom = True
        hfmodel.cpu()
        del hfmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mglogits = mgmodel(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except torch.cuda.OutOfMemoryError:
            print('oom for megatron model forward')
            is_oom = True
        mgmodel.cpu()
        del mgmodel

    epsilon = 1e-5
    for idx, (hfh, mgh) in enumerate(zip(hf_hiddens, mg_hiddens)):
        assert len(hfh) == len(mgh)
        for k, hfv in hfh.items():
            mgv, hfv = mgh[k].cpu(), hfv.cpu()
            same_num = (hfv != mgv).sum()
            diff_num = ((hfv - mgv) > epsilon).sum()
            diff_max = (hfv - mgv).abs().max()
            print(f'layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hfv.numel()}] diff_max:{diff_max}')

    if not is_oom:
        same_num = (hflogits != mglogits).sum()
        diff_num = ((hflogits - mglogits) > epsilon).sum()
        diff_max = (hflogits - mglogits).abs().max()
        print(f'logits: {same_num}, diff>{epsilon}:[{diff_num}/{hflogits.numel()}] diff_max:{diff_max}')

def add_extra_args(parser):
    parser = get_args(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path)

    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

    # 初期シードを設定
    model_parallel_cuda_manual_seed(42)  # 任意のシード値を使用

    # RNGトラッカーを取得
    rng_tracker = get_cuda_rng_tracker()
    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)

    mg_model = load_megatron_model(args.mg_ckpt_path)
    
    check_hf_mg_forward(hf_model, mg_model, args)

if __name__ == "__main__":
    main()