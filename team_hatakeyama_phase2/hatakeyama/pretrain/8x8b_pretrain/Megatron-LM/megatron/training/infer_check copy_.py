# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import gc
import dataclasses
from datetime import datetime
import math
import logging
import os
import sys
from .log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
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

from .utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model)
from .global_vars import (
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

stimer = StragglerDetector()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch_distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


import requests


def check_maintenance_event() -> bool:
    # GCP metadata request URL
    URL = "http://metadata.google.internal/computeMetadata/v1/instance/maintenance-event"
    headers = {"Metadata-Flavor": "Google"}

    try:
        response = requests.get(URL, headers=headers)
        # status code check
        if response.status_code == 200:
            event = response.text
            if "TERMINATE_ON_HOST_MAINTENANCE" in event:
                print("INFO: Host maintenance detected.")
                return True
            else:
                return False
        else:
            print("request is failed. status code:", response.status_code)
            return False
    except Exception as e:
        print("exception occurs when requesting:", e)
        return False


def num_floating_point_operations(args, batch_size):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    return (
        12
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (
                    1
                    + (args.num_query_groups / args.num_attention_heads)
                    + (args.seq_length / args.hidden_size)
                ) * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Logit.
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )


def append_to_progress_log(string):
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, 'a') as f:
            job_id = os.getenv('SLURM_JOB_ID', '')
            num_gpus = args.world_size
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                    f"# GPUs: {num_gpus}\t{string}\n")


def get_start_time_from_progress_log():
    """
    Gets start time of earliest job with same world size. Also returns the number
    of floating-point operations completed in last saved checkpoint.
    """
    args = get_args()
    assert args.save is not None
    progress_log_filename = os.path.join(args.save, "progress.txt")

    # start_time is time when job with same world size started.
    # start_num_floating_point_operations is the number of floating-point operations
    # completed when this job started.
    # latest_num_floating_point_operations is the number of floating-point operations
    # completed in most recent saved checkpoint.
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0

    def _get_field(string, type):
        return type(string.split(': ')[1])

    with open(progress_log_filename, 'r') as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split('\t')
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = \
                    _get_field(line_tokens[7], float)
            if world_size_in_line != args.world_size:
                # Re-start search if we see a different world size.
                start_time = None
                start_num_floating_point_operations = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = \
                        latest_num_floating_point_operations
    assert start_time is not None and start_num_floating_point_operations is not None, \
        "Should have seen at least one 'Starting job' entry with same world_size"
    return datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'), \
        start_num_floating_point_operations


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.double,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    one_logger = get_one_logger()
    if one_logger:
        one_logger.log_metrics({
            'train_iterations_warmup': 5
        })

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    config = get_model_config(model[0])

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup'], barrier=True)

    print(model)

    prefix = f'calculation megatron model'

    evaluate_and_print_results(prefix, model, process_non_loss_data_func, config,
                                verbose=True, write_to_tensorboard=not args.skip_train)


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        config = get_model_config(model[0])
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size)
        model = [DDP(config,
                     ddp_config,
                     model_chunk,
                     data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                     expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()

    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                       scale_lr_cond, lr_mult)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None or args.pretrained_checkpoint is not None:
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler)
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler

def compute_throughputs_and_append_to_progress_log(iteration,
                                                   num_floating_point_operations_so_far):
    args = get_args()
    if args.save is None:
        return

    # Compute job throughput.
    # args.num_floating_point_operations_so_far keeps track of floating-point operations
    # completed at the start of job.
    global _TRAIN_START_TIME
    job_throughput = \
        (num_floating_point_operations_so_far -
         args.num_floating_point_operations_so_far) / (
            (time.time() - _TRAIN_START_TIME) * 10**12 * args.world_size)

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = \
        (num_floating_point_operations_so_far -
         start_num_floating_point_operations) / (
            elapsed_time * 10**12 * args.world_size)

    tokens_so_far = args.consumed_train_samples * args.seq_length

    append_to_progress_log(f"Saved checkpoint\tIteration: {iteration}\t"
                           f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
                           f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
                           f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
                           f"Tokens (in billions): {tokens_so_far / 10**9:.2f}")

def evaluate(model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True)
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * eval_num_microbatches

    timers('evaluate').stop()
    timers.log(['evaluate'])

    return total_loss_dict, collected_non_loss_data, False


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    
    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose)
    # Timelimit hit during evaluation
    if timelimit:
        return
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)
        if wandb_writer and is_last_rank():
            wandb_writer.log({
                '{} validation'.format(key): total_loss_dict[key].item()},
                iteration)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def check_hf_mg_forward(hfmodel, mgmodel, mgargs):

    input_ids = torch.tensor([[11644,   8506,  22564,  27608,  43188,   4344, 12395,  1991,  29554, 36689]]).long().cuda()
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

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    hf_model = AutoModelForCausalLM.from_pretrained(args.load)
    mg_model = model_provider()
    convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    check_hf_mg_forward(hf_model, mg_model, args)
    
if __name__ == "__main__":
    main()