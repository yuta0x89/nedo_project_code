#作ったモデルを動かしてみる

"""
srun --nodelist=slurm0-a3-ghpc-[0] --gpus-per-node=1 --time=30-00:00:00 -c 16 --pty bash -i
source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
conda activate share-jk_py310_TEv1.7_FAv2.5.7
python play_model.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time
def perplexity(model, tokenizer, text) -> torch.Tensor:
    tokenized_input = tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model(tokenized_input, labels=tokenized_input)
    ppl = torch.exp(output.loss)
    return ppl.item()


model_path="/storage5/shared/Llama-3-8/0719cleaned_tp1-pp4-ct1-LR5.0E-5-MINLR0.5E-5-WD0.1-WARMUP500-nnodes16/hf_iter_0006400"
print("begin loading model")
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")

tokenizer_name="/storage5/shared/Llama-3-8/HF/cont_0126000_lr1_5e_m4"
tokenizer = AutoTokenizer.from_pretrained(model_path)


pipe=pipeline('text-generation',model=model,tokenizer=tokenizer, max_new_tokens=200, repetition_penalty=1.2)
text_list=["今日はいい",
"富士山は",
"質問: 今日の天気は? 回答:",
]

for text in text_list:
    perp=perplexity(model,tokenizer,text)
    s_time=time.time()
    res=pipe(text)[0]["generated_text"]
    consumed_time=time.time()-s_time
    print("-------")
    print("input: ", text)
    print("perplexity: ",perp)
    print("time: ", consumed_time)
    print("time/character: ", consumed_time/len(res))
    print("output: ",res)

while True:
    text=input()
    res=pipe(text)[0]["generated_text"]
    print(res)