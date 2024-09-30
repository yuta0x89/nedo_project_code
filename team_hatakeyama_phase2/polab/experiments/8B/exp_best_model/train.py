import logging
from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig
from datasets import disable_caching, load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments
from trl import DPOTrainer
from trainer import GPOTrainer, GPOConfig

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)

import os
import sys
import argparse

import yaml
import numpy as np
import re

disable_caching()

logger = logging.getLogger(__name__)

@dataclass
class DataArgments:
    exp_config_path: str
    
    def __post_init__(self):
        with open(self.exp_config_path, 'r') as file:
            exp_config = yaml.safe_load(file)
        self.data_config_list = exp_config['data']
        self.template_config = exp_config['template']

@dataclass
class WandbArguments:
    wandb_entity: str
    wandb_project: str
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[str] = None

@dataclass
class GPOTrainingArguments:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: Optional[list[str]] = None
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
        
        if self.peft_target_model == "llama-mini":
            self.peft_target_modules = [
                "q_proj",
                "v_proj",
            ]
        if self.peft_target_model == "llama":
            # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
            self.peft_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        if self.peft_target_model == "llama-all":
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

class Preprocessor:
    """サンプルにさまざまな前処理ステップを適用するクラス。
    Attributes:
        template_config (dict): テンプレートの設定。
        tokenizer (Tokenizer): 使用するトークナイザのインスタンス。
        result_key (str): 結果を保存するキー。
    """
    
    def __init__(self, template_config, tokenizer):
        """指定されたテンプレート設定、トークナイザ、および結果キーでPreprocessorを初期化する。
        Args:
            template_config (dict): テンプレートの設定。
            tokenizer (Tokenizer): 使用するトークナイザのインスタンス。
        """
        self.template_config = template_config
        self.tokenizer = tokenizer

    def add_bos(self, sample):
        """promptにbosトークンを追加する。
        Args:
            sample (dict): 処理するデータを含むサンプル。
        Returns:
            dict: BOSトークンが追加された更新済みのサンプル。
        """
        if sample is not None:
            sample['prompt'] = self.tokenizer.bos_token + sample['prompt']
            return sample
        else:
            return None

    def cleanup_text(self, text):
        """テキストからノイズになり得る要素を削除する。
        Args:
            text (str): 処理するテキスト。
        Returns:
            str: ノイズを除いたテキスト。
        """
        noisy_words = [self.tokenizer.bos_token, self.tokenizer.eos_token]
        pattern = '|'.join(re.escape(word) for word in noisy_words)
        return re.sub(pattern, '', text)
    
    def preprocess_trl(self, sample, prompt='prompt', chosen='chosen', rejected='rejected'):
        """指定されたキーに基づいて指示、好ましい回答、好ましくない回答をもつサンプルを前処理する。
        Args:
            sample (dict): 処理するデータを含むサンプル。
            prompt (str): サンプル内の指示部分のキー。デフォルトは'prompt'。
            chonsen (str, optional): サンプル内の好ましい回答のキー。デフォルトは'prompt'。
            rejected (str): サンプル内の好ましくない回答のキー。デフォルトは'rejected'。
        Returns:
            dict: テンプレートが適用された更新済みのサンプル。
        """
        sample['prompt'] = self.template_config['system'] + self.template_config['instruction'] + sample[prompt] + self.template_config['output']
        if chosen != 'chosen':
            sample['chosen'] = sample.pop(chosen)
        if rejected != 'rejected':
            sample['rejected'] = sample.pop(rejected)
        # bos, eosトークンが入っていれば削除
        for key in ['prompt', 'chosen', 'rejected']:
            sample[key] = self.cleanup_text(sample[key])
        return sample

    def preprocess_trl_with_score(self, sample, prompt='prompt', response1='response1', score1='score1', response2='response2', score2='score2'):
        """指定されたキーに基づいて指示と、回答とスコア(高い方が良い回答)の組を2つ持つサンプルを前処理する。
        Args:
            sample (dict): 処理するデータを含むサンプル。
            prompt (str): サンプル内の指示部分のキー。デフォルトは'prompt'。
            response1 (str, optional): サンプル内の1つ目の回答のキー。デフォルトは'response1'。
            score1 (str): サンプル内の1つ目の回答のスコアのキー。デフォルトは'score1'。
            response2 (str, optional): サンプル内の1つ目の回答のキー。デフォルトは'response2'。
            score2 (str): サンプル内の1つ目の回答のスコアのキー。デフォルトは'score2'。
        Returns:
            dict: テンプレートが適用された更新済みのサンプル。
        """
        if isinstance(sample['score1'], (int, float)) and isinstance(sample['score2'], (int, float)):
            sample['prompt'] = self.template_config['system'] + self.template_config['instruction'] + sample[prompt] + self.template_config['output']
            if sample['score1'] > sample['score2']:
                chosen = sample['response1']
                rejected = sample['response2']            
            else:
                chosen = sample['response2']
                rejected = sample['response1']
            sample['chosen'] = chosen
            sample['rejected'] = rejected
            # bos, eosトークンが入っていれば削除
            for key in ['prompt', 'chosen', 'rejected']:
                sample[key] = self.cleanup_text(sample[key])
            return sample

    def preprocess_openai_messages(self, sample, messages, role='role', chosen='chosen', rejected='rejected', content='content', add_system_message=True, ignore_original_system_message=True, add_eos=False):
        """サンプル内のOpenAIメッセージを前処理する。
        Args:
            sample (dict): 処理するデータを含むサンプル。
            messages (str): サンプル内のメッセージのキー。
            role (str): メッセージ内の役割のキー。デフォルトは'role'。
            content (str): メッセージ内のコンテンツのキー。デフォルトは'content'。
            chonsen (str, optional): サンプル内の好ましい回答のキー。デフォルトは'prompt'。
            rejected (str): サンプル内の好ましくない回答のキー。デフォルトは'rejected'。
            add_system_message (bool): システムメッセージを追加するかどうか。デフォルトはTrue。
            ignore_original_system_message (bool): 元のシステムメッセージを無視するかどうか。デフォルトはTrue。
            add_eos (bool): eosトークンを各ターンの終わりに追加するかどうか。デフォルトはFalse。
        Returns:
            dict: 前処理されたメッセージを含む更新済みのサンプル。
        """
        messages = sample[messages]
        turns = sum([1 if 'user' in m[role] or 'human' in m[role] else 0 for m in messages])
        turn = 0
        result = []
        for m in messages:
            if 'system' in m[role]:
                if not ignore_original_system_message:
                    # bos, eosトークンが入っていれば削除
                    system_text = self.cleanup_text(m[content])
                    result.append(system_text)
            elif 'user' in m[role] or 'human' in m[role]:
                turn += 1
                # bos, eosトークンが入っていれば削除
                instruction_text = self.cleanup_text(m[content])
                result.append(self.template_config['instruction'] + instruction_text + self.template_config['output'])
            else:
                # bos, eosトークンが入っていれば削除
                response = self.cleanup_text(m[content])
                if turn < turns and add_eos:
                    response += self.tokenizer.eos_token 
                result.append(response)
        result = ''.join(result)
        if add_system_message:
            result = self.template_config['system'] + result
        sample['prompt'] = result

        if chosen != 'chosen':
            sample['chosen'] = sample.pop(chosen)
        if rejected != 'rejected':
            sample['rejected'] = sample.pop(rejected)
        # bos, eosトークンが入っていれば削除、promptはすでにeosトークンが挿入されている場合があるので対象外
        for key in ['chosen', 'rejected']:
            sample[key] = self.cleanup_text(sample[key])
        return sample

    def apply_preprocessing(self, sample, steps):
        """サンプルに前処理ステップのシーケンスを適用する。
        Args:
            sample (dict): 処理するデータを含むサンプル。
            steps (list): 前処理関数の名前と引数を含む辞書のリスト。
        Returns:
            dict: すべての前処理ステップが適用された後の更新済みのサンプル。
        """
        for step in steps:
            func = getattr(self, step['name'])
            args = step.get('args', {})  # argsがない場合は空の辞書を使用
            sample = func(sample, **args)
        return sample


def load_datasets(data_list, template_config, cache_dir, tokenizer, split):
    datasets = []
    dataset_info = {}
    preprocessor = Preprocessor(template_config, tokenizer)
    
    for data_config in data_list:
        if split in data_config['split']:
            dataset = load_dataset(data_config['name'], split=data_config['split'][split])
            processed_samples = [preprocessor.apply_preprocessing(sample, data_config['preprocess']) for sample in dataset]
            processed_samples = [sample for sample in processed_samples if sample is not None]
            dataset = Dataset.from_dict({key: [sample[key] for sample in processed_samples] for key in processed_samples[0].keys()})
            print(dataset.select_columns(['prompt', 'chosen', 'rejected'])[:1])
            datasets.append(dataset)
            dataset_info[data_config['name']] = len(dataset)

    dataset_info['total'] = sum(dataset_info.values())
    print('=' * 30 + '\n' +  f'{split} data info:\n' + '\n'.join([f' {k}: {v} samples' for k, v in dataset_info.items()]) + '\n' + '=' * 30)
    
    return concatenate_datasets(datasets)


def main() -> None:
    parser = HfArgumentParser((GPOConfig, GPOTrainingArguments, WandbArguments, DataArgments))
    training_args, gpo_training_args, wandb_args, data_args = parser.parse_args_into_dataclasses()
    
    tokenizer_name_or_path: str = (
        gpo_training_args.tokenizer_name_or_path or gpo_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    logger.info(training_args)
    logger.info(gpo_training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=gpo_training_args.use_fast,
        additional_special_tokens=gpo_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #tokenizer.add_special_tokens({'additional_special_tokens': ['\n']})
    #tokenizer.add_prefix_space=False
    logger.info("Loading data")

    # キャッシュファイルの保存先
    model_cache_dir = gpo_training_args.hf_cache_dir + '/hub'
    dataset_cache_dir = gpo_training_args.hf_cache_dir + '/datasets'

    data_config_list = data_args.data_config_list
    template_config = data_args.template_config
    # yamlの改行コードは変換されて読みこまれてしまうため、正しく認識されるように再度変換を行う
    template_config = {k: v.replace('\\n', '\n') for k, v in template_config.items()}

    train_dataset = load_datasets(data_config_list, template_config, dataset_cache_dir, tokenizer, 'train')
    if training_args.do_eval:
        print("do eval")
        eval_dataset = load_datasets(data_config_list, template_config, dataset_cache_dir, tokenizer, 'eval')
        
    else:
        eval_dataset = None

    logger.info(f"Loading model from {gpo_training_args.model_name_or_path}")
    kwargs = gpo_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({gpo_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    print(kwargs)

    if gpo_training_args.model_name_or_path.find("mergoo")>=0:
        from mergoo.models.modeling_llama import LlamaForCausalLM
        #aceclerateをつかったマルチgpuでの学習がうまくいかなかった
        print("init mergoo model")

        model = LlamaForCausalLM.from_pretrained(
            gpo_training_args.model_name_or_path, 
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
            gpo_training_args.model_name_or_path,
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
    if gpo_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=gpo_training_args.peft_lora_r,
            target_modules=gpo_training_args.peft_target_modules,
            lora_alpha=gpo_training_args.peft_lora_alpha,
            lora_dropout=gpo_training_args.peft_lora_dropout,
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
        wandb.init(entity=wandb_args.wandb_entity, project=wandb_args.wandb_project,
                    group=wandb_args.wandb_group, name=wandb_args.wandb_name,
                    tags=wandb_args.wandb_tags.split(','))
    
        # 実験スクリプトをアーティファクトとして保存
        save_scripts = gpo_training_args.save_scripts.split(',') + [os.path.abspath(__file__)]
        artifact = wandb.Artifact(name='scripts', type='code')
        for script_path in save_scripts:
            artifact.add_file(script_path)
        wandb.log_artifact(artifact)

    logger.info("Setting up trainer")
    print(training_args)
    
    trainer = GPOTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        ropo_alpha=training_args.ropo_alpha,
        ropo_gamma=training_args.ropo_gamma
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