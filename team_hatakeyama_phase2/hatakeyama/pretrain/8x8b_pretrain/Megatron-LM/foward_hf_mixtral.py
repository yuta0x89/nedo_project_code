import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/storage5/shared/Nishijima/Llama-3-8b-MoE/3rd_tonyu_iter_800",
    #"/storage5/shared/Llama-3-8/HF/cont_0111000",
    #"/storage5/shared/hatakeyama/0706moe_abeja/mergekit/model_four",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    max_position_embeddings=19,
)

model.model.layers = model.model.layers[:8]
print(model)

model.eval()
with torch.no_grad():
    token_ids = torch.tensor([[        1, 20811,   349,   396, 13126,   369, 13966,   264, 120, 2000, 25000,
                        54930, 4903, 55, 100, 30432, 56821, 20040, 60003]])
    labels = torch.tensor([[    1, 20811,   349,   396, 13126,   369, 13966,   264, 120, 2000, 25000,
                        54930, 4903, 55, 100, 30432, 56821, 20040, 60003]])
    attention_mask = torch.ones_like(token_ids)
    position_ids = attention_mask.long().cumsum(-1) - 1

    model_inputs = {
        'input_ids': token_ids.to(model.device),
        'labels': labels.to(model.device),
        'past_key_values': None,
        'use_cache': True,
        'position_ids': position_ids.to(model.device),
        'attention_mask': attention_mask.to(model.device),
        'output_attentions': True,
        'output_hidden_states': True,
        #'output_router_logits': True,
        'return_dict': True
    }
    outputs = model(**model_inputs)

    print(outputs.logits)
    
    import numpy as np

    # NumPy 配列が hflogits[0] に格納されていると仮定
    numpy_array = outputs.logits.cpu().numpy()[0]

    # CSV ファイルとして保存
    output_path = "/storage5/shared/Nishijima/test_moe/3rd_tonyu_iter_800_3rd_hf_8layers4.npy"
    np.save(output_path, numpy_array)

    print(f"np file ファイルが保存されました: {output_path}")