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
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, ModelConfig, SFTTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import wandb

disable_caching()

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: list[str] = None
    response_template: str = None
    eval_data_files: Optional[list[str]] = None
    instruction_template: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = False
    additional_special_tokens: Optional[list[str]] = None
    max_seq_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    peft_lora_r: int = 8

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


def load_datasets(data_files):
    datasets = []
    for data_file in data_files:
        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]
        #dataset = dataset.select_columns("text")
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def main() -> None:
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    tokenizer_name_or_path: str = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )

    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    logger.info("Loading data")

    #train_dataset = load_datasets(sft_training_args.data_files)
    dolly_en_dataset = load_dataset("databricks/databricks-dolly-15k")
    dolly_en_dataset = dolly_en_dataset['train'].filter(lambda example: example['context'] == '').select(range(1000))
    #dolly_en_dataset = dolly_en_dataset.filter(lambda example: len(example['instruction']) <= 1800)
    column_renames = {
        'instruction': 'text',
        'response': 'output'
    }
    # データセットに変更を適用
    for old_name, new_name in column_renames.items():
        dolly_en_dataset = dolly_en_dataset.rename_column(old_name, new_name)

    #install dolly ja
    dolly_ja_dataset = load_dataset("kunishou/databricks-dolly-15k-ja")['train']
    #dolly_ja_dataset = dolly_ja_dataset['train'].filter(lambda example: example['input'] == '')
    column_renames = {
        'instruction': 'text'
    }
    # データセットに変更を適用
    for old_name, new_name in column_renames.items():
        dolly_ja_dataset = dolly_ja_dataset.rename_column(old_name, new_name)

    #install ichikara
    ichikara_dataset = load_dataset("p1atdev/ichikara-instruction", '20231221-003')['train']
    #ichikara_dataset = ichikara_dataset['train'].filter(lambda example: len(example['text']) <= 1800)
    
    #merge datasets
    datasets =  [dolly_ja_dataset, dolly_en_dataset , ichikara_dataset]
    train_dataset = concatenate_datasets(datasets)
    print(train_dataset)

    print(f'endoftext is {tokenizer.eos_token}')

    # プロンプトの生成
    def generate_prompt(example):
        messages = [
            {
                'role': "system",
                'content': "あなたは優秀なAIアシスタントです。"
            },
            {
                'role': "user",
                'content': example["text"]
            },
            {
                'role': "assistant",
                'content': example["output"]
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # textカラムの追加
    def add_text(example):
        example["content"] = generate_prompt(example)
        return example

    dataset = dataset.map(add_text)
    dataset = dataset.remove_columns(["text", "output", "category", "input", "index", "context", "ID"])
    
    
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files)
        training_args.do_eval = True
    else:
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({sft_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )
    

    model_config = ModelConfig(
        model_name_or_path=sft_training_args.model_name_or_path,
        attn_implementation=sft_training_args.use_flash_attention_2 #flash attention cause some problems.
    )
    model_kwargs = dict(
        trust_remote_code=model_config.trust_remote_code,
        #attn_implementation=model_config.attn_implementation,
        #torch_dtype=torch.bfloat16,  #flash attention cause some problems.
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

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

    logger.info("Setting up trainer")
    #training_args.report_to="wandb"
    #wandb.init(project="my-test-project", name="yelp_review_full bert classification")
    
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="content",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=sft_training_args.max_seq_length,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
