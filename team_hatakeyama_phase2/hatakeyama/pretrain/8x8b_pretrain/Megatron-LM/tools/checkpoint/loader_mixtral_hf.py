# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import types
import torch
import transformers
from tqdm import tqdm

import os
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = "1"

def add_arguments(parser):
    group = parser.add_argument_group(title='Mixtral 8x7B HF loader.')

    group.add_argument(
        '--true-vocab-size', type=int, default=None,
        help='original size of vocab, if specified will trim padding from embedding table.'
    )
    group.add_argument(
        '--vocab-file', type=str, default=None,
        help='Path to the vocab file. If specified will use this to get vocab size and '
        'trim padding from the embedding table.'
    )
    group.add_argument(
        '--tokenizer-model', required=True,
        help='Sentencepiece tokenizer model.'
    )
    group.add_argument(
        '--megatron-path', type=str, default="/storage5/Megatron-LM",
        help='Base directory of deepspeed repository'
    )
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
    if major < 4 or minor < 31:
        raise ValueError("the version transformers should greater or equal 4.31")


def load_args_from_checkpoint(args):

    # Read Mixtral args.
    args_path = os.path.join(args.load, "config.json")
    with open(args_path) as f:
        mixtral_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 2048
    args.sequence_parallel = True
    args.max_position_embeddings = mixtral_args.get("max_position_embeddings", 32768)
    args.num_experts = mixtral_args.get("num_local_experts", None)
    args.num_experts_per_tok = mixtral_args.get("num_experts_per_tok", 2)
    args.moe_type = "mixtral"
    args.hidden_size = mixtral_args["hidden_size"]
    args.num_attention_heads = mixtral_args["num_attention_heads"]
    args.num_layers = mixtral_args["num_hidden_layers"]
    args.global_batch_size = 1024
    args.norm_epsilon = mixtral_args["rms_norm_eps"]
    args.iteration = 1
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    #args.position_embedding_type = "rope"
    args.swiglu = True
    args.tokenizer_type = "Llama2Tokenizer"
    args.rope_theta = mixtral_args["rope_theta"]
    #args.params_dtype = mixtral_args["torch_dtype"]
    args.bf16 = True
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = True
    args.vocab_size = mixtral_args["vocab_size"]
    args.padded_vocab_size = mixtral_args["vocab_size"]
    args.mixtral = mixtral_args
    args.ffn_hidden_size = mixtral_args["intermediate_size"]
    args.gradient_accumulation_fusion = False
    #args.tensor_model_parallel_size=4

    if "num_key_value_heads" in mixtral_args:
        args.group_query_attention = True
        args.num_query_groups = mixtral_args["num_key_value_heads"]


def set_preprocess_state(args, model, hf_model):
    """Set embedding params."""
    model.language_model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    """Set output layer & norm params."""
    model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)
    model.language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_attn_state(args, layer, hf_layer):
    """Set self-attention params."""

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size  # 常に1
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention else args.num_attention_heads) // tp
    dim = args.kv_channels
    if not nh % ng == 0:
        raise ValueError("nh % ng should equal 0")

    # Copy weights (re-order dimensions for Megatron).
    attn.query_key_value.weight.data.copy_(torch.cat([
        hf_attn.q_proj.weight.reshape((ng, dim * nh // ng, -1)),
        hf_attn.k_proj.weight.reshape((ng, dim, -1)),
        hf_attn.v_proj.weight.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.dense.weight.data.copy_(hf_attn.o_proj.weight)


def set_mlp_state(args, layer, hf_layer):
    """Set Mixtral SMOE params."""

    mlp = layer.mlp
    hf_mlp = hf_layer.block_sparse_moe

    #gate = mlp.gate.weight.data.copy_
    gate = mlp.router.weight.data.copy_
    hf_gate_weight = hf_mlp.gate.weight
    gate(hf_gate_weight)

    #print(f"local_expert :{mlp.local_experts}")
    #print(f"local_expert.dense_h_to_4h :{mlp.local_experts[0]}")
    #print(f"local_expert.dense_h_to_4h :{mlp.local_experts[0].dense_h_to_4h.weight.shape}")
    #print(f"local_expert.dense_4h_to_h :{mlp.local_experts[0].dense_4h_to_h.weight.shape}")
    #print(f"hf_mlp.experts W1: {hf_mlp.experts[0].w1.weight.shape}")
    #print(f"hf_mlp.experts W2: {hf_mlp.experts[0].w2.weight.shape}")
    #print(f"hf_mlp.experts W3: {hf_mlp.experts[0].w3.weight.shape}")

    for idx in range(args.num_experts):
        local_expert = mlp.local_experts[idx]
        hf_expert = hf_mlp.experts[idx]

        # W1とW3 (dense_h_to_4h)の重みをコピー
        combined_weight = torch.cat([hf_expert.w1.weight, hf_expert.w3.weight], dim=0)
        local_expert.dense_h_to_4h.weight.data.copy_(combined_weight)

        # W2 (dense_4h_to_h)の重みをコピー
        local_expert.dense_4h_to_h.weight.data.copy_(hf_expert.w2.weight)

        if local_expert.dense_h_to_4h.bias is not None:
            combined_bias = torch.cat([hf_expert.w1.bias, hf_expert.w3.bias])
            local_expert.dense_h_to_4h.bias.data.copy_(combined_bias)


def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer params."""

    layer = model.language_model.encoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)
    layer.input_norm.weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.post_attention_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def initialize_rng_states(args):
    # RNGトラッカーの初期化
    from megatron.core.tensor_parallel import random as mpu_random
    mpu_random.initialize_rng_tracker()
    
    # データ並列RNGトラッカー名を取得し、追加
    data_parallel_rng_name = mpu_random.get_data_parallel_rng_tracker_name()
    rng_tracker = mpu_random.get_cuda_rng_tracker()
    rng_tracker.add(data_parallel_rng_name, args.seed)

    # 専門家並列RNGトラッカー名を取得し、追加（MoEを使用する場合）
    if args.num_experts > 0:
        expert_parallel_rng_name = mpu_random.get_expert_parallel_rng_tracker_name()
        rng_tracker.add(expert_parallel_rng_name, args.seed + 1)

def load_checkpoint_to_model(args):
    """Set model params."""
    from pretrain_gpt import model_provider
    from transformers import AutoModelForCausalLM, AutoModel

    # RNG状態の初期化
    initialize_rng_states(args)

    # Load Huggingface model.
    print(f'start loading model')
    hf_model = AutoModelForCausalLM.from_pretrained(args.load,
    #hf_model = AutoModel.from_pretrained(args.load, 
                                        torch_dtype=args.params_dtype,
                                        low_cpu_mem_usage=True,
                                        device_map="cpu")

    print(f'finish loading model')
    print(hf_model)

    # Init Megatron model.
    #args.use_mcore_models=True
    #args.attention_softmax_in_fp32=True
    #args.expert_parallel_size=1
    #import torch.distributed as dist
    #import torch
    #from megatron.training.initialize import initialize_megatron
    #from megatron.core import parallel_state
    #from megatron.core.tensor_parallel import random as mpu_random

    #def init_distributed(args):
    #    from megatron.training.arguments import parse_args
    #    import argparse
    #    # 分散バックエンドの初期化
    #    torch.distributed.init_process_group(backend='nccl')
    #    
    #    # initialize_megatron関数を呼び出し
    #    from megatron.training.initialize import initialize_megatron        
    #    initialize_megatron(args)
    #    
    #    # モデル並列グループの初期化
    #    parallel_state.initialize_model_parallel(
    #        tensor_model_parallel_size=args.tensor_model_parallel_size,
    #        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    #        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
    #        expert_model_parallel_size=args.expert_model_parallel_size
    #    )
    #    
    #    # RNGトラッカーの初期化
    #    mpu_random.initialize_rng_tracker()
    #    
    #    # モデル並列CUDAの乱数シードを設定
    #    mpu_random.model_parallel_cuda_manual_seed(args.seed)
    #    
    #    # データ並列RNGトラッカー名を取得
    #    data_parallel_rng_name = mpu_random.get_data_parallel_rng_tracker_name()
    #    
    #    # CudaRNGStatesTrackerを取得し、データ並列RNGを追加
    #    rng_tracker = mpu_random.get_cuda_rng_tracker()
    #    rng_tracker.add(data_parallel_rng_name, args.seed)

    #    # 専門家並列RNGトラッカー名を取得し、追加（MoEを使用する場合）
    #    expert_parallel_rng_name = mpu_random.get_expert_parallel_rng_tracker_name()
    #    rng_tracker.add(expert_parallel_rng_name, args.seed + 1)

    # モデル作成前にこの関数を呼び出す
    #init_distributed(args)
    #"""
    #def init_distributed():
    #    if torch.cuda.is_available():
    #        if not dist.is_initialized():
    #            dist.init_process_group(
    #                backend='nccl',
    #                world_size=1,
    #                rank=0
    #            )
    #            #dist.init_process_group(backend='nccl')
    #            print(f"Distributed initialized: {dist.is_initialized()}")

    #init_distributed()
    #"""

    # Initialize Megatron
    #from megatron.training.initialize import initialize_megatron
    #initialize_megatron(args)

    # Set up the model parallel groups
    #from megatron.core import mpu
    #mpu.initialize_model_parallel(args.tensor_model_parallel_size, 
    #                            args.pipeline_model_parallel_size,
    #                            args.virtual_pipeline_model_parallel_size,
    #                            args.expert_model_parallel_size)

    model = model_provider(True, True).to(args.params_dtype)

    # Set model state.
    set_preprocess_state(args, model, hf_model)
    set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(queue, args):

    # Mixtral requires HF transformers >=4.36.0.
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
        from megatron.legacy.model.enums import ModelType
        from megatron.legacy import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path.")
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
    # ここのtensor_model_parallel_sizeはダミーなので注意
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    margs.use_mcore_models = False
    margs.transformer_impl = args.loader_transformer_impl
    if args.loader_transformer_impl == 'transformer_engine':
        margs.attention_softmax_in_fp32 = True

    margs.expert_model_parallel_size=1

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
    check_for_arg('sliding_window_size', 0)

    # Determine how to make our models.
    if not args.model_type == 'GPT':
        raise ValueError("Mixtral is a GPT model.")
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
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.moe_type = margs.moe_type
    md.num_experts = margs.num_experts
    md.num_experts_per_tok = margs.num_experts_per_tok
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
    md.true_vocab_size = None
    md.make_vocab_size_divisible_by = int(128 / args.target_tensor_parallel_size * 2)
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.rope_theta = margs.rope_theta
    md.sequence_parallel = 1
    margs.add_position_embedding = False
    # 本家じゃないtokenizerを使う時の対応
    tokenizer_model_name = margs.tokenizer_model.split("/")[-1]
    if tokenizer_model_name != "Mixtral-8x7B-v0.1":
        from megatron.training.tokenizer import build_tokenizer
        import argparse
        tokenizer_args = {
            "tokenizer_type": "SentencePieceTokenizer",
            "tokenizer_model": margs.tokenizer_model
        }
        tokenizer_args = argparse.Namespace(**tokenizer_args)
        tokenizer_args.rank = 1
        tokenizer_args.make_vocab_size_divisible_by = md.make_vocab_size_divisible_by
        tokenizer_args.tensor_model_parallel_size = 1  # dummy
        tokenizer_args.vocab_extra_ids = 0
        hf_tokenizer  = build_tokenizer(tokenizer_args)
        md.true_vocab_size = hf_tokenizer.vocab_size

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
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
        if hasattr(model.language_model.embedding, 'position_embeddings'):
            raise ValueError("model should have position_embeddings")

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.language_model.encoder.layers[layer_num]
        message["input norm weight"] = layer.input_norm.weight.data
        message["post norm weight"] = layer.post_attention_norm.weight.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.dense.bias.data
            message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_gate_weight = []
        mlp_gate_bias = []

        layer = model.language_model.encoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.query_key_value.weight.data)
        dense_weight.append(layer.self_attention.dense.weight.data)
        #mlp_gate_weight.append(layer.mlp.gate.weight.data)
        mlp_gate_weight.append(layer.mlp.router.weight.data)

        if md.linear_bias:
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            #mlp_gate_bias.append(layer.mlp.gate.bias.data)
            mlp_gate_bias.append(layer.mlp.router.bias.data)

        for expert_idx in range(margs.num_experts):

            message[f"mlp {expert_idx} w1, w2 weight"] = getattr(layer.mlp.local_experts, f"{expert_idx}").dense_h_to_4h.weight.data
            message[f"mlp {expert_idx} w3 weight"] = getattr(layer.mlp.local_experts, f"{expert_idx}").dense_4h_to_h.weight.data
            if md.linear_bias:
                message[f"mlp {expert_idx} w1, 22 bias"] = getattr(layer.mlp.local_experts, f"{expert_idx}").dense_h_to_4h.bias.data
                message[f"mlp {expert_idx} w2 bias"] = etattr(layer.mlp.local_experts, f"{expert_idx}").dense_4h_to_h.bias.data

        # Simple concat of the rest.
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp gate weight"] = torch.cat(mlp_gate_weight, dim=1)

        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            message["mlp gate bias"] = torch.cat(mlp_gate_bias, dim=0)

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
    except:
        queue.put("exit")
        raise