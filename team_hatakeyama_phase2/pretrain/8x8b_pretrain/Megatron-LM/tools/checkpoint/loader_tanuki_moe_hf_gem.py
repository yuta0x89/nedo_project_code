# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import transformers
from tqdm import tqdm
import types


def add_arguments(parser):
    group = parser.add_argument_group(title='Tanuki-MoE HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')
    
    group.add_argument(
        "--target-tensor-parallel-size", type=int, default=4,
        help="Target tensor model parallel size, defaults to the tensor parallel size "
        "in the input checkpoint if provided by the loader, otherwise to 1"
    )
    parser.add_argument(
        "--tensor-model-parallel-size", type=int, default=4,
        help="Target tensor model parallel size, defaults to the tensor parallel size "
        "in the input checkpoint if provided by the loader, otherwise to 1"
    )
    parser.add_argument('--bf16', action='store_true', help='Whether to load weights in bf16.')
    group.add_argument('--loader-transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 31


def load_args_from_checkpoint(args):

    # Read Llama args.
    tanuki_args_path = os.path.join(args.load, "config.json")
    with open(tanuki_args_path) as f:
        tanuki_args = json.load(f)

    # Update Megatron args.
    args.seq_length = tanuki_args["max_position_embeddings"]
    args.max_position_embeddings = tanuki_args["max_position_embeddings"]
    args.hidden_size = tanuki_args["hidden_size"]
    args.num_attention_heads = tanuki_args["num_attention_heads"]
    args.num_layers = tanuki_args["num_hidden_layers"]
    args.global_batch_size = 1024
    args.norm_epsilon = tanuki_args["rms_norm_eps"]
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.tokenizer_type = "Llama2Tokenizer"
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = True
    args.vocab_size = tanuki_args["vocab_size"]
    args.padded_vocab_size = tanuki_args["vocab_size"]
    args.tanuki = tanuki_args
    args.ffn_hidden_size = tanuki_args["intermediate_size"]
    args.rope_theta = tanuki_args.get("rope_theta", 10000.0)
    args.params_dtype = tanuki_args["torch_dtype"]
    args.num_experts = tanuki_args["num_local_experts"]
    args.num_experts_per_tok = tanuki_args["num_experts_per_tok"]
    args.bf16 = True
    args.fp16 = False

    if "num_key_value_heads" in tanuki_args:
        args.group_query_attention = True
        args.num_query_groups = tanuki_args["num_key_value_heads"]


def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    model.language_model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)
    model.language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.query_key_value.weight.data.copy_(torch.cat([
        hf_attn.q_proj.weight.reshape((ng, dim * nh // ng, -1)),
        hf_attn.k_proj.weight.reshape((ng, dim, -1)),
        hf_attn.v_proj.weight.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.dense.weight.data.copy_(hf_attn.o_proj.weight)


#def set_mlp_state(args, layer, hf_layer):
#    '''Set MLP params.'''

#    mlp = layer.mlp
#    hf_mlp = hf_layer.mlp

#    mlp.dense_h_to_4h.weight.data.copy_(torch.cat([
#        hf_mlp.gate_proj.weight,
#        hf_mlp.up_proj.weight,
#    ], dim=0))
#    mlp.dense_4h_to_h.weight.data.copy_(hf_mlp.down_proj.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.language_model.encoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)
    layer.input_norm.weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.post_attention_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)

def set_mlp_state(args, layer, hf_layer):
    mlp = layer.mlp
    hf_mlp = hf_layer.block_sparse_moe

    #print(layer)
    #print(layer.mlp)

    mlp.router.weight.data.copy_(hf_mlp.gate.weight)

    #print(mlp.local_experts)

    for i in range(args.num_experts):
        expert = mlp.local_experts[i]
        #expert = getattr(mlp.local_experts, f"expert_{i}")
        hf_expert = hf_mlp.experts[i]
        
        expert.dense_h_to_4h.weight.data.copy_(torch.cat([
            hf_expert.w1.weight,
            hf_expert.w3.weight,
        ], dim=0))
        expert.dense_4h_to_h.weight.data.copy_(hf_expert.w2.weight)

def load_checkpoint_to_model(args):
    '''Set model params.'''

    from pretrain_gpt import model_provider
    from transformers import LlamaForCausalLM, MixtralForCausalLM

    # Load Huggingface model.
    hf_model = MixtralForCausalLM.from_pretrained(
        args.load,
        torch_dtype=args.params_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    print(f'finish loading model')
    print(hf_model)

    # Init Megatron model.
    #from megatron.core.tensor_parallel.random import _set_cuda_rng_state
    #_set_cuda_rng_state("data-parallel-rng")
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

    # 初期シードを設定
    model_parallel_cuda_manual_seed(42)  # 任意のシード値を使用

    # RNGトラッカーを取得
    rng_tracker = get_cuda_rng_tracker()

    # データ並列RNG状態を設定
    #rng_tracker.add("data-parallel-rng")

    model = model_provider(True, True).to(args.params_dtype)
    print(model)

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(queue, args):

    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    margs.target_tensor_parallel_size = args.target_tensor_parallel_size
    margs.target_moedel_parallel_size = args.target_pipeline_parallel_size
    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    margs.expert_parallel_size=1

    margs.use_mcore_models = False
    margs.transformer_impl = args.loader_transformer_impl
    if args.loader_transformer_impl == 'transformer_engine':
        margs.attention_softmax_in_fp32 = True

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('expert_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)
    check_for_arg('num_experts')
    check_for_arg('num_experts_per_tok')

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'Tanuki is a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder
    margs.params_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    fused_kernels.load(margs)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.true_vocab_size = None  # skips padding in saver
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.rope_theta = margs.rope_theta
    md.num_experts = margs.num_experts
    md.num_experts_per_tok = margs.num_experts_per_tok

    # 本家じゃないtokenizerを使う時の対応
    margs.add_position_embedding = False
    print(f"margs: {margs}")
    #tokenizer_model_name = margs.tokenizer_model.split("/")[-1]
    #if tokenizer_model_name != "Mixtral-8x7B-v0.1":
    #    from megatron.training.tokenizer import build_tokenizer
    #    import argparse
    #    tokenizer_args = {
    #        "tokenizer_type": "HFTokenizer",
    #        "tokenizer_model": margs.tokenizer_model
    #    }
    #    tokenizer_args = argparse.Namespace(**tokenizer_args)
    #    tokenizer_args.rank = 1
    #    tokenizer_args.make_vocab_size_divisible_by = md.make_vocab_size_divisible_by
    #    tokenizer_args.tensor_model_parallel_size = 1  # dummy
    #    tokenizer_args.vocab_extra_ids = 0
    #    hf_tokenizer  = build_tokenizer(tokenizer_args)
    #    md.true_vocab_size = hf_tokenizer.vocab_size


    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.language_model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.language_model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.language_model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.language_model.encoder.layers[layer_num]
        message["input norm weight"] = layer.input_norm.weight.data
        message["post norm weight"] = layer.post_attention_norm.weight.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.dense.bias.data
            #message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        router_weight = []
        router_bias = []
        expert_weights = [[] for _ in range(md.num_experts)]
        expert_biases = [[] for _ in range(md.num_experts)]

        layer = model.language_model.encoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.query_key_value.weight.data)
        dense_weight.append(layer.self_attention.dense.weight.data)
        router_weight.append(layer.mlp.router.weight.data)

        if md.linear_bias:
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)

        for expert_idx in range(md.num_experts):
            #print(f"layer.mlp {layer.mlp}")
            #expert = getattr(layer.mlp.experts, f"expert_{i}")
            expert = layer.mlp.local_experts[expert_idx]
            #expert_weights[i].append(expert.dense_h_to_4h.weight.data)
            message[f"mlp {expert_idx} weight1"] = expert.dense_h_to_4h.weight.data
            #expert_weights[i].append(expert.dense_4h_to_h.weight.data)
            message[f"mlp {expert_idx} weight2"] = expert.dense_4h_to_h.weight.data
        
        if md.linear_bias:
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            router_bias.append(layer.mlp.router.bias.data)
            for i in range(md.num_experts):
                expert = getattr(layer.mlp.experts, f"expert_{i}")
                #expert_biases[i].append(expert.dense_h_to_4h.bias.data)
                message[f"mlp {expert_idx} bias1"] = expert.dense_h_to_4h.bias.data
                #expert_biases[i].append(expert.dense_4h_to_h.bias.data)
                message[f"mlp {expert_idx} bias1"] = expert.dense_4h_to_h.bias.data
        # Simple concat of the tensors.
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["router weight"] = torch.cat(router_weight, dim=0)
        #for i in range(md.num_experts):
        #    message[f"mlp {expert_idx} weight1"] = getattr(layer.mlp.local_experts, f"{expert_idx}").dense_h_to_4h.weight.data
        #    message[f"mlp {expert_idx} weight2"] = getattr(layer.mlp.local_experts, f"{expert_idx}").dense_4h_to_h.weight.data
            
        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            message["router bias"] = torch.cat(router_bias, dim=0)
            for i in range(md.num_experts):
                message[f"expert_{i}_bias"] = [torch.cat(expert_biases[i][0], dim=0), torch.cat(expert_biases[i][1], dim=0)]

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": model.language_model.encoder.final_norm.weight.data,
    }
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": model.language_model.output_layer.weight.data
        }
        queue_put("output layer", message)

    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception as e:
        queue.put("exit")
        raise e
