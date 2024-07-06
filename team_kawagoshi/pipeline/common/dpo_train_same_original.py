import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import disable_caching, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)

disable_caching()

logger = logging.getLogger(__name__)

@dataclass
class DPOTrainingArguments:
    model_name_or_path: str
    checking_collator: bool = False
    use_fast: bool = False
    additional_special_tokens: Optional[list[str]] = None
    tokenizer_name_or_path: Optional[str] = None
    max_seq_length: int = 2048
    max_prompt_length: int = 1024
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    beta: float = 0.3
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 128
    peft_lora_alpha: int = 256
    peft_lora_dropout: float = 0.05
    use_wandb: bool =False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tag: Optional[str] = None
    label_smoothing: float = 0.0
    precompute_ref_log_probs: bool = False

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ]
            elif self.peft_target_model == "llama-all":
                # https://note.com/kan_hatakeyama/n/ncd09c52d26c7
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs

def return_prompt_and_responses(samples) -> dict[str, list[str]]:
    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []

    for conversation, chosen, rejected in zip(
        samples["conversations"], samples["chosen"], samples["rejected"]
    ):
        prompt: str = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
        for utterance in conversation:
            if utterance["from"] == "human":
                prompt += f"\n\n### 指示:\n{utterance['value']}"
            else:
                prompt += f"\n\n### 応答:\n{utterance['value']}"
        prompt += "\n\n### 応答:\n"
        prompts.append(prompt)
        chosens.append(chosen)
        rejecteds.append(rejected)

    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


def main():
    parser = HfArgumentParser((TrainingArguments, DPOTrainingArguments))
    training_args, dpo_training_args = parser.parse_args_into_dataclasses()
    # DeepSpeedの設定が適用されているかを確認
    #print("Training arguments:")
    #for arg in vars(training_args):
    #    print(f"{arg}: {getattr(training_args, arg)}")
    
    tokenizer_name_or_path: str = (
        dpo_training_args.tokenizer_name_or_path or dpo_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=dpo_training_args.use_fast,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading the paired dataset")
    dataset = load_dataset("llm-jp/hh-rlhf-12k-ja", split="train", num_proc=4)
    original_columns = dataset.column_names
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=4,
        remove_columns=original_columns,
    )
    shuffled_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = shuffled_dataset["train"]
    eval_dataset = shuffled_dataset["test"]
    logger.info(f"Loaded {len(train_dataset)} samples for training")
    logger.info(f"Loaded {len(eval_dataset)} samples for evaluation")

    logger.info(f"Loading model from {dpo_training_args.model_name_or_path,}")
    kwargs = dpo_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({dpo_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        dpo_training_args.model_name_or_path,
        use_cache=False,
        trust_remote_code=True,
        **kwargs,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        dpo_training_args.model_name_or_path,
        use_cache=False,
        trust_remote_code=True,
        **kwargs,
    )

    #training_args.remove_unused_columns=False
    #training_args.optim="adamw_torch"
    #training_args.torch_compile=True

    peft_config: Optional[LoraConfig] = None
    if dpo_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=dpo_training_args.peft_lora_r,
            target_modules=["q_proj","v_proj"],
            #target_modules=["c_attn", "c_proj", "c_fc"],
            lora_alpha=dpo_training_args.peft_lora_alpha,
            lora_dropout=dpo_training_args.peft_lora_dropout,
            fan_in_fan_out=True,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    if dpo_training_args.use_wandb:
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        is_local_rank_0 = (torch.distributed.get_rank() % get_accelerator().device_count() == 0) if torch.distributed.is_initialized() else True
        if is_local_rank_0:
            import socket
            import wandb
            logger.info("Setting up wandb")
            wandb.init(entity=dpo_training_args.wandb_entity, project=dpo_training_args.wandb_project,
                       group=dpo_training_args.wandb_group, name=socket.gethostname(),
                       tags=[dpo_training_args.wandb_tag] if dpo_training_args.wandb_tag else None)

    logger.info("Setting up trainer")

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=dpo_training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=dpo_training_args.max_seq_length,
        max_prompt_length=dpo_training_args.max_prompt_length,
        max_target_length=dpo_training_args.max_seq_length - dpo_training_args.max_prompt_length,
    )

    logger.info("Training")
    dpo_trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
