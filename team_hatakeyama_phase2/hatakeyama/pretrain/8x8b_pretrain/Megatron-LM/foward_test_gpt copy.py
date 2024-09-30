# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

import torch
from torch import Tensor
import torch.distributed as torch_distributed
import wandb

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config, StragglerDetector
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.pipeline_parallel import get_forward_backward_func

from megatron.training import pretrain, get_model
from megatron.training.checkpointing import _load_base_checkpoint

from megatron.training.utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model)
from megatron.training.global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
    update_global_dynamic_checkpoint,
    get_global_dynamic_checkpoint,
    set_maintenance_detected_time,
    get_maintenance_detected_time
)

import torch.distributed as dist
import logging

def setup_rank_logger():
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"rank_{rank}.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def sync_all_ranks(message):
    if dist.is_initialized():
        logger.info(f"Waiting at sync point: {message}")
        dist.barrier()
        logger.info(f"Passed sync point: {message}")
    else:
        logger.info(f"Sync point (single process): {message}")

from megatron.core.models.gpt import GPTModel

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )
    else:
        assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    print("Start forward")
    print("Entering forward_step function")
    print(f"Model type: {type(model)}")
    print(f"Data iterator type: {type(data_iterator)}")
    sync_all_ranks("Before forward step")
    args = get_args()
    timers = get_timers()

    # BATCH SIZE PER SINGLE GPU MUST BE ONE!!!!
    tokens = torch.tensor([[    1, 20811,   349,   396, 13126,   369, 13966,   264]], device=torch.cuda.current_device())
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=torch.cuda.current_device())
    loss_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], device=torch.cuda.current_device())
    attention_mask = torch.tensor([[[[False,  True,  True,  True,  True,  True,  True,  True],
              [False, False,  True,  True,  True,  True,  True,  True],
              [False, False, False,  True,  True,  True,  True,  True],
              [False, False, False, False,  True,  True,  True,  True],
              [False, False, False, False, False,  True,  True,  True],
              [False, False, False, False, False, False,  True,  True],
              [False, False, False, False, False, False, False,  True],
              [False, False, False, False, False, False, False, False]]]], device=torch.cuda.current_device())
    #labels = torch.tensor([[32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000]], device=torch.cuda.current_device())
    labels = torch.tensor([[20896, 26570, 20896, 21876, 25931, 25931, 20896, 20896]], device=torch.cuda.current_device())

    print("Start calc")
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    print("finish calc")
    if isinstance(output_tensor, tuple):
        print("!")
        return list(output_tensor), partial(loss_func, loss_mask)
    else:
        print("?")
        sync_all_ranks("After forward step")
        return output_tensor, partial(loss_func, loss_mask)


if __name__ == "__main__":
    # Initalize and get arguments, timers, and Tensorboard writer.
    # メイン処理の開始時にこれを呼び出す
    logger = setup_rank_logger()

    # 以下のようにして使用する
    logger.info("Starting forward pass")
    logger.info("Completed forward pass")
    logger.info("Before print statement")
    initialize_megatron(extra_args_provider=None, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    sync_all_ranks("After initialization")
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder

    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

    # 初期シードを設定
    model_parallel_cuda_manual_seed(42)  # 任意のシード値を使用

    # RNGトラッカーを取得
    rng_tracker = get_cuda_rng_tracker()

    model = get_model(model_provider, ModelType.encoder_or_decoder)
    print_rank_0(model)
    sync_all_ranks("After model setup")

    config = get_model_config(model[0])

    if args.load is not None:
        load_dir = getattr(args, 'load')
        state_dict, _, _ = _load_base_checkpoint(load_dir, rank0=False)
        model[0].module.load_state_dict(state_dict['model'], strict=True)
    
    print("loading Done")

    for model_module in model:
        model_module.eval()
        #model_module.train()

    print("Evalmode Done")
    total_loss_dict = {}

    #for model_chunk in model:
    #    model_chunk.zero_grad_buffer(zero_buffer=(not args.use_distributed_optimizer))

    forward_backward_func = get_forward_backward_func()

    #import numpy as np

    # NumPy 配列が hflogits[0] に格納されていると仮定
    #numpy_array = hflogits[0]

    # CSV ファイルとして保存
    #output_path = "/storage5/shared/Nishijima/test_moe/step4900_hf.csv"
    #np.savetxt(output_path, numpy_array, delimiter=",")

    #print(f"CSV ファイルが保存されました: {output_path}")

    print("into forward_backward_func")
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=None,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=True)
    sync_all_ranks("After forward_backward_func")

    print("before print")
    #print(model[0].module.module.language_model.encoder.layers[-1].mlp.experts[0].w1.weight)
    #print(model[0].module.module.language_model.encoder.layers[-1].mlp.gate.weight.main_grad)

    print_rank_last(losses_reduced)
    sync_all_ranks("After printing results")
    