import torch
import torch.distributed as dist
from megatron.training import get_args, get_timers, print_rank_0
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.initialize import initialize_megatron
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
import logging
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger('gpt_inference')

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm),
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

    return model

def prepare_data():
    tokens = torch.tensor([[1, 20811, 349, 396, 13126, 369, 13966, 264]], device=torch.cuda.current_device())
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=torch.cuda.current_device())
    attention_mask = torch.tensor([[[[False, True, True, True, True, True, True, True],
                                     [False, False, True, True, True, True, True, True],
                                     [False, False, False, True, True, True, True, True],
                                     [False, False, False, False, True, True, True, True],
                                     [False, False, False, False, False, True, True, True],
                                     [False, False, False, False, False, False, True, True],
                                     [False, False, False, False, False, False, False, True],
                                     [False, False, False, False, False, False, False, False]]]], device=torch.cuda.current_device())
    labels = torch.tensor([[20896, 26570, 20896, 21876, 25931, 25931, 20896, 20896]], device=torch.cuda.current_device())
    return tokens, position_ids, attention_mask, labels

def forward_step(data_iterator, model):
    args = get_args()
    tokens, position_ids, attention_mask, labels = next(data_iterator)

    # テンソル並列ランクに基づいてデータをスライス
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    
    # vocab_parallel_rankに基づいてtokensとlabelsをスライス
    slice_size = tokens.size(-1) // tp_size
    start_idx = tp_rank * slice_size
    end_idx = (tp_rank + 1) * slice_size
    tokens = tokens[..., start_idx:end_idx].contiguous()
    labels = labels[..., start_idx:end_idx].contiguous()

    # position_idsとattention_maskはスライスしない
    
    # モデルに入力を渡す
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
  
    return output_tensor, None
    
def run_inference():
    args = get_args()
    timers = get_timers()

    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)

    if args.load is not None:
        load_checkpoint(model, None, None)

    # モデルを評価モードに設定
    for model_module in model:
        model_module.eval()

    # 勾配計算を無効化
    with torch.no_grad():
        forward_backward_func = get_forward_backward_func()

        data = prepare_data()
        data_iterator = iter([data])

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=args.seq_length,
            micro_batch_size=1
        )

    if mpu.is_pipeline_last_stage():
        print_rank_0(f'Inference result: {losses_reduced}')

if __name__ == "__main__":
    initialize_megatron(extra_args_provider=None, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder
    #args.tensor_model_parallel_size = 2  # TP=4
    #args.pipeline_model_parallel_size = 4  # PP=2

    model_parallel_cuda_manual_seed(args.seed)

    run_inference()