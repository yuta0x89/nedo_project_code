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
import wandb

disable_caching()

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: list[str] = None
    sample_sizes: list[int] = None
    checking_collator: bool = False
    response_template: str = None
    eval_data_files: Optional[list[str]] = None
    instruction_template: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = False
    additional_special_tokens: Optional[list[str]] = None
    max_seq_length: int = 2048
    valid_ratio: float = 0.1
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
    use_wandb: bool =False
    #neftune_noise_alpha: Optional[float] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tag: Optional[str] = None

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


from datasets import load_dataset, concatenate_datasets

def load_datasets(data_files, sample_sizes):
    datasets = []
    base = "/storage2/ucllm_nedo_prod/train/dataset/SFT/"
    for data_file, sample_size in zip(data_files, sample_sizes):
        name = base + data_file        
        dataset = load_dataset("json", data_files=name)#, cache_dir="/storage2/tempcache")
        #except FileNotFoundError:
        #    print(f"File {name} not found.")
        #    continue
        dataset = dataset["train"]
        if sample_size > 1:
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        #if "text" in dataset.column_names:
        dataset = dataset.select_columns("text")
        datasets.append(dataset)
        print(f"loaded {data_file}, number {sample_size}")
    return concatenate_datasets(datasets)



def main() -> None:
    #HF_DATASETS_CACHE="/storage2"
    import os
    
    #cache_dir = os.makedirs("/storage2/tempcache",exist_ok=True)
    #os.environ['TRANSFORMERS_CACHE'] = "/storage2/tempcache"
    #os.environ['HF_DATASETS_CACHE'] = "storage2/tempcache"
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

    train_dataset = load_datasets(sft_training_args.data_files, sft_training_args.sample_sizes)

    #logger.info(f"fillter the dataset the length is less {sft_training_args.max_seq_length -3}")
    #print(f"original dataset size : {train_dataset.num_rows}")
    #train_dataset = train_dataset.filter(lambda x: len(tokenizer.encode(x['text'])) < sft_training_args.max_seq_length -3)
    #print(f"filltered dataset size : {train_dataset.num_rows}")
    
    print(train_dataset)
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files)
        training_args.do_eval = True
    else:        
        train_test_split = train_dataset.train_test_split(test_size=sft_training_args.valid_ratio, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

    print(f'endoftext is {tokenizer.eos_token}')

    # response_templateは必須指定
    instruction_template = tokenizer.encode("\n\n### 指示:\n", add_special_tokens=False)[1:]
    response_template = tokenizer.encode("\n\n### 応答:\n", add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer)

    def print_tokens_with_ids(txt):
        tokens = tokenizer.tokenize(txt, add_special_tokens=False)
        token_ids = tokenizer.encode(txt, add_special_tokens=False)
        print(list(zip(tokens, token_ids)))

    prompt = """あなたはAIアシスタントです。 #\n\n### 指示:\n今日の天気を教えてください。\n\n### 応答\n今日の天気は晴れです。</s>"""
    print_tokens_with_ids(prompt)  # [..., ('▁Hello', 15043), ('<0x0A>', 13), ('<0x0A>', 13), ('##', 2277), ('#', 29937), ('▁Ass', 4007), ('istant', 22137), (':', 29901), ...]
    
    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({sft_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
    )

    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )
    
    #model_config = ModelConfig(
    #    model_name_or_path=sft_training_args.model_name_or_path,
    #    attn_implementation=sft_training_args.use_flash_attention_2 #flash attention cause some problems.
    #)
    #model_kwargs = dict(
    #    trust_remote_code=model_config.trust_remote_code,
    #    #attn_implementation=model_config.attn_implementation,
    #    #torch_dtype=torch.bfloat16,  #flash attention cause some problems.
    #)
    #model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

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
    
    if sft_training_args.use_wandb:
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        is_local_rank_0 = (torch.distributed.get_rank() % get_accelerator().device_count() == 0) if torch.distributed.is_initialized() else True
        if is_local_rank_0:
            import socket
            import wandb
            logger.info("Setting up wandb")
            wandb.init(entity=sft_training_args.wandb_entity, project=sft_training_args.wandb_project,
                       group=sft_training_args.wandb_group, name=socket.gethostname(),
                       tags=[sft_training_args.wandb_tag] if sft_training_args.wandb_tag else None)

    logger.info("Setting up trainer")
    #training_args.report_to="wandb"
    #wandb.init(project="my-test-project", name="yelp_review_full bert classification")
    
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=sft_training_args.max_seq_length
    )

    if sft_training_args.checking_collator:
        logger.info("Checking collator")
        from torch.utils.data import DataLoader
        loader = DataLoader(trainer.train_dataset, collate_fn=collator, batch_size=32)

        batch = next(iter(loader))
        for i in range(32):
            print(batch['labels'][i])
            print(batch['input_ids'][i])
            print(tokenizer.decode(batch['input_ids'][i]))


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

    #export HF_HOME="/storage2/tempcache"
    #export TRION_CACHE_DIR="/storage2/tempcache/trion"
    #export TORCH_EXTENSIONS_DIR="/storage2/tempcache/torch"