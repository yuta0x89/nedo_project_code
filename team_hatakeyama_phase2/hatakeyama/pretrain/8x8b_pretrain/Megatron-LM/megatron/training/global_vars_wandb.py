# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""
import argparse
import os
import sys
import torch
import typing

from megatron.training import dist_signal_handler
from megatron.core import Timers
from megatron.training.tokenizer import build_tokenizer
from .microbatches import build_num_microbatches_calculator

_GLOBAL_WANDB_WRITER = None

def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER

import torch
import wandb
from datetime import datetime
import socket

def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER, 'wandb writer')
    
    # ローカルランクを計算
    local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
    
    # 各ノードのローカルランク0のみがW&Bを初期化
    if getattr(args, 'wandb_project', '') and local_rank == 0:
        if args.wandb_name is None:
            raise ValueError("Please specify the wandb experiment name!")

        now = datetime.now().strftime("%m-%d-%H-%M")
        hostname = socket.gethostname()  # ホスト名を取得
        print(f"hostname {hostname}")
        exp_name = f"{args.wandb_name}-{hostname[-2:]}-{now}"

        wandb_kwargs = {
            'entity': args.wandb_entity,
            'name': exp_name,
            'project': args.wandb_project,
            'config': vars(args)}

        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb

def set_global_variables_wandb(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _set_wandb_writer(args)

#def get_maintenance_detected_time() -> float | None:
from typing import Union

def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)
