import logging
from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig
from datasets import disable_caching, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import os
import sys
import argparse

import yaml

disable_caching()

logger = logging.getLogger(__name__)


@dataclass
class DataArgments:
    exp_config_path: str
    data_list: list[str] = None

    def __post_init__(self):
        with open(self.exp_config_path, 'r') as file:
            exp_config = yaml.safe_load(file)
        self.data_list = exp_config['data']

@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    response_template: str
    instruction_template: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: Optional[list[str]] = None
    max_seq_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llama-all"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    param_neftune_noise_alpha: float = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    save_scripts: Optional[str] = None
    
    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError(
                "load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            logger.warning(
                f"you should se the peft_target_modules when using peft_target_model"
            )
        if self.peft_target_model == "mergoo":
            logger.info("Setting LORA for Mergoo")
            self.peft_target_modules = [
                #"gate_proj",
                "gate_proj.gate",
            ]
        if self.peft_target_model == "mixtral":
            logger.info("Setting LORA for mixtral")
            self.peft_target_modules = [
                "gate",
            ]



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


def get_preprocess_func(preprocess_config, tokenizer):
    template = preprocess_config['template']
    keys = preprocess_config['keys']

    if template == 'chat_template':
        def preprocess_func(example):
            return tokenizer.apply_chat_template(example[keys['messages']], tokenize=False)
        return preprocess_func
    elif template == "Alpaca":
        def preprocess_func(example):
            prompt = f"以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{example[keys['instruction']]}"
            if 'input' in keys and type(example[keys['input']]) is str and len(example[keys['input']]) > 0:
                prompt += f"\n\n### 入力:\n{example[keys['input']]}"
            prompt += f"\n\n### 応答:\n{example[keys['output']]}"
            return prompt
        return preprocess_func
    elif template == "Alpaca_OpenAI_messages":
        def preprocess_func(example):
            messages = example[keys['messages']]
            result = []
            for m in messages:
                if 'user' in m['role'] or 'human' in m['role']:
                    result.append("\n\n### 指示:\n{}\n\n### 応答:\n".format(m['content']))
                else:
                    result.append(m['content'])
            return '以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。' + ''.join(result)
        return preprocess_func
    else:
        raise ValueError(f'Unknown preprocess template: "{template}"')


def load_datasets(data_list, cache_dir, tokenizer, split):

    datasets = []
    dataset_info = {}
    
    for data_config in data_list:
        """
        train_list=[]
        with open(data_file,"r") as f:
            for line in f:
                train_list.append(json.loads(line))
        data = {key: [dic[key] for dic in train_list] for key in train_list[0]}
        dataset=Dataset.from_dict(data)
        """
        if split in data_config['split']:
            dataset = load_dataset(data_config['name'], split=data_config['split'][split], cache_dir=cache_dir)
            preprocess_func = get_preprocess_func(data_config['preprocess'], tokenizer)
            dataset = dataset.map(lambda example: {"text": preprocess_func(example)})
            dataset = dataset.select_columns("text")
            print(dataset['text'][:1])
            datasets.append(dataset)
            dataset_info[data_config['name']] = len(dataset)
    
    dataset_info['total'] = sum(dataset_info.values())
    print('=' * 30 + '\n' +  f'{split} data info:\n' + '\n'.join([f' {k}: {v} samples' for k, v in dataset_info.items()]) + '\n' + '=' * 30)
    
    return concatenate_datasets(datasets)


def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments, DataArgments))
    training_args, sft_training_args, data_args = parser.parse_args_into_dataclasses()
    
    tokenizer_name_or_path: str = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    logger.info(training_args)
    logger.info(sft_training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #tokenizer.add_special_tokens({'additional_special_tokens': ['\n']})
    #tokenizer.add_prefix_space=False
    logger.info("Loading data")

    # キャッシュファイルの保存先
    model_cache_dir = sft_training_args.hf_cache_dir + '/hub'
    dataset_cache_dir = sft_training_args.hf_cache_dir + '/dataset'

    #train_dataset = load_datasets(sft_training_args.data_files)
    train_dataset = load_datasets(data_args.data_list, dataset_cache_dir, tokenizer, 'train')
    if training_args.do_eval:
        print("do eval")
        #eval_dataset = load_datasets(sft_training_args.eval_data_files)
        eval_dataset = load_datasets(data_args.data_list, dataset_cache_dir, tokenizer, 'eval')
        training_args.do_eval = True
        #training_args.evaluation_strategy = "steps"
        #training_args.eval_steps= 50

    else:
        eval_dataset = None

    logger.info("Formatting prompts")
    #自動でtokenを設定すると､うまくいかないことがある
    response_ids = tokenizer.encode(
        sft_training_args.response_template, add_special_tokens=False)[1:]
    instruction_ids = tokenizer.encode(
        sft_training_args.instruction_template, add_special_tokens=False)[1:]

    #手動で設定する
    logger.info("manually setting template ids")

    #response_ids=[5092, 272, 1045, 2850, 327]  # 指示､応答に対するtokenを手動で設定
    #instruction_ids=[5092, 272, 3994, 327]     #
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_ids, response_template=response_ids, tokenizer=tokenizer
    )
    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({sft_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    print(kwargs)

    if sft_training_args.model_name_or_path.find("mergoo")>=0:
        from mergoo.models.modeling_llama import LlamaForCausalLM
        #aceclerateをつかったマルチgpuでの学習がうまくいかなかった
        print("init mergoo model")

        model = LlamaForCausalLM.from_pretrained(
            sft_training_args.model_name_or_path,
            cache_dir=model_cache_dir,
            #device_map="auto",  #accelerateの場合はoffにする
            **kwargs,
        )# 'gate' / router layers are untrained hence loaded warning would appeare for them

        # train only router (gating) layers
        if False:
            n_weights, n_router_weights  = 0,0
            for name, weight in model.named_parameters():
                if "gate" not in name:
                    weight.requires_grad_(False)
                    n_router_weights += 1
                n_weights += 1
            print("train params:")
            print(n_weights)
            print(n_router_weights)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            sft_training_args.model_name_or_path,
            cache_dir=model_cache_dir,
            trust_remote_code=True,
            #device_map="auto",
        #device_map={'':device_string},

            **kwargs,
        )
    #if True:
    #    for name, weight in model.named_parameters():
    #        if "gate" not in name and "weight" in name:
    #            weight.requires_grad_(False)

    peft_config: Optional[LoraConfig] = None
    if sft_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=sft_training_args.peft_lora_r,
            target_modules=sft_training_args.peft_target_modules,
            lora_alpha=sft_training_args.peft_lora_alpha,
            lora_dropout=sft_training_args.peft_lora_dropout,
            fan_in_fan_out=True,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if training_args.gradient_checkpointing:
            for param in model.parameters():
                param.requires_grad = False
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    # ノードごとにwandb.init()を実行してログを送信
    # 標準コード参考（https://github.com/hotsuyuki/llm-jp-sft/blob/ucllm_nedo_dev_v20240415.1.0/train.py#L178-L198）
    import deepspeed
    from deepspeed.accelerator import get_accelerator
    # ローカルランクが0のときにW&Bをセットアップ
    is_local_rank_0 = (torch.distributed.get_rank() % get_accelerator().device_count() == 0) if torch.distributed.is_initialized() else True
    if is_local_rank_0:
        import socket
        import wandb
        logger.info("Setting up wandb")
        wandb.init(entity=sft_training_args.wandb_entity, project=sft_training_args.wandb_project,
                    group=sft_training_args.wandb_group, name=sft_training_args.wandb_name,
                    tags=sft_training_args.wandb_tags.split(','))

    
        # 実験スクリプトをアーティファクトとして保存
        save_scripts = sft_training_args.save_scripts.split(',') + [os.path.abspath(__file__)]
        artifact = wandb.Artifact(name='scripts', type='code')
        for script_path in save_scripts:
            artifact.add_file(script_path)
        wandb.log_artifact(artifact)

    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=sft_training_args.max_seq_length,
        neftune_noise_alpha=sft_training_args.param_neftune_noise_alpha,  # NEFTune https://qiita.com/m__k/items/23ced0db6846e97d41cd
    )

    logger.info("Training")
    if training_args.resume_from_checkpoint:
        print("load checkpoint")
        trainer.train(resume_from_checkpoint='/storage5/EvalPractice/model/2_0524with_halcination_little_codes_-storage5-llm-models-hf-step62160_fin_inst_0524with_halcination_little_codes-inst_parquet_lr_5e-5/checkpoint-7600')
    else:
        print("start from zero")
        trainer.train()

    logger.info("Saving model")
    trainer.save_model()


if __name__ == "__main__":
    logging.basicConfig(
        #level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
