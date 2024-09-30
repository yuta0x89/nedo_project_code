import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
import argparse

class CustomNormalization(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, logits):
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True)
        return (logits - mean) / (std + 1e-5)

class ModifiedMixtralModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(
            config._name_or_path,
            attn_implementation="eager",
            config=config
        )
        self.modify_with_custom_norm()

    def modify_with_custom_norm(self):
        for layer in self.model.model.layers:
            if hasattr(layer, 'block_sparse_moe'):
                for expert in layer.block_sparse_moe.experts:
                    original_w1 = expert.w1
                    expert.w1 = nn.Sequential(
                        original_w1,
                        CustomNormalization(original_w1.out_features)
                    )
            else:
                print(f"Warning: Unexpected layer structure. Cannot modify layer: {layer}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs['attn_implementation'] = "eager"
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

def load_and_modify_model(model_name):
    print(f"Loading model: {model_name}")
    config = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").config
    config._name_or_path = model_name
    return ModifiedMixtralModel(config)

def save_model_and_tokenizer(model, tokenizer, output_dir):
    print(f"Saving modified model and tokenizer to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save custom configuration
    custom_config = model.config.to_dict()
    custom_config['attn_implementation'] = "eager"
    custom_config['architectures'] = ["ModifiedMixtralModel"]
    
    import json
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(custom_config, f)

def main():
    parser = argparse.ArgumentParser(description="Modify and save Mixtral model with custom normalization")
    parser.add_argument("--load", type=str, default="mistralai/Mixtral-8x7B-v0.1",
                        help="Model to load and modify")
    parser.add_argument("--save", type=str, required=True,
                        help="Directory to save the modified model")
    args = parser.parse_args()

    # モデルの読み込みと修正
    model = load_and_modify_model(args.load)

    # トークナイザーの読み込み
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    # 修正したモデルとトークナイザーの保存
    save_model_and_tokenizer(model, tokenizer, args.save)

    print("Model modification and saving completed.")

    # 保存したモデルの読み込みと使用例
    print("\nTesting the saved model:")
    loaded_model = ModifiedMixtralModel.from_pretrained(args.save)
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.save)

    input_text = "日本の観光名所は？"
    input_ids = loaded_tokenizer.encode(input_text, return_tensors="pt")
    output = loaded_model.generate(input_ids, max_length=50)
    print(loaded_tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()