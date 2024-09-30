import os, json, datetime
import random
import torch
import numpy as np

import pandas as pd
from datetime import timezone, timedelta
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


fix_seed()
login(token=os.environ.get(HUGGING_FACE_API_KEY))

os.makedirs('./out', exist_ok=True)
json_file_path = os.path.join('./out', 'aya_ja_evol-calm3-22b.jsonl')
skip_file_path = os.path.join('./out', 'aya_ja_evol-calm3-22b.txt')
progress_file_path = os.path.join('./out', 'progress.txt')

temperature = 0.0
top_p = 1.0
max_tokens = 2048

dataset = load_dataset("ryota39/aya_ja_evol")['train']
df = pd.DataFrame(dataset)

# model_path = 'cyberagent/calm3-22b-chat'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# llm = LLM(
#     model=model_path,
#     dtype="bfloat16",
#     download_dir="./ws"
# )

model_path = 'team-hatakeyama-phase2/tanuki-8B-someya-exp007'
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    download_dir="./ws"
)

sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
)


def load_last_processed_index(progress_file_path):
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r', encoding='utf-8') as progress_file:
            last_index = progress_file.read().strip()
            return int(last_index) if last_index.isdigit() else -1
    return -1

def save_last_processed_index(progress_file_path, last_index):
    with open(progress_file_path, 'w', encoding='utf-8') as progress_file:
        progress_file.write(str(last_index))

def generate_responses(model, prompts, sampling_params):
    responses = model.generate(prompts, sampling_params)
    return responses

def build_prompt(record):
    messages = [
        {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
        {"role": "user", "content": f'''{record['prompt']}'''}
    ]

    prompt = ""
    for message in messages:
        prompt += f"{message['role']}: {message['content']}\n"
    prompt += "assistant: "

    return prompt

def process_batches(dataframe, batch_size=256):
    total_rows = len(dataframe)
    last_processed_index = load_last_processed_index(progress_file_path)

    start_index = last_processed_index + 1
    if start_index >= total_rows:
        print("All rows are already processed.")
        return

    processed_count = start_index

    idx = 0
    while processed_count < total_rows:
        batch_end_index = min(processed_count + batch_size, total_rows)
        batch_df = dataframe.iloc[processed_count:batch_end_index]
        prompts = []
        indices = []
        rows = []

        for index, row in batch_df.iterrows():
            prompt = build_prompt(row)
            prompts.append(prompt)
            indices.append(index)
            rows.append(row)

        try:
            generated_responses = generate_responses(
                llm,
                prompts,
                sampling_params
                )

            # バッチ内でイテレーションして応答を処理
            for local_idx, response, prompt, row in zip(indices, generated_responses, prompts, rows):
                if response.outputs and len(response.outputs) > 0:
                    response = response.outputs[0].text.strip()

                    data = {
                        'idx': idx,
                        'prompt': prompt,
                        # 'chosen': response,
                        'rejected': response,
                        'chosen_model': 'cyberagent/calm3-22b-chat',
                        'rejected_model': 'team-hatakeyama-phase2/tanuki-8B-exp007',
                    }

                    with open(json_file_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                else:
                    print(f"No outputs for prompt: {prompt}")
                    with open(skip_file_path, 'a', encoding='utf-8') as skip_file:
                        skip_file.write(f"{local_idx}\n")
                idx += 1
        except Exception as e:
            print(f"Error: {e}")
            with open(skip_file_path, 'a', encoding='utf-8') as skip_file:
                skip_file.write(f"{','.join(map(str, indices))}\n")

        # バッチごとにプロンプトとインデックスを初期化
        prompts = []
        indices = []
        rows = []

        # 最後に処理したインデックスを保存
        last_index = batch_end_index - 1
        save_last_processed_index(progress_file_path, last_index)

        processed_count = batch_end_index

process_batches(df, batch_size=512)

print("Finished processing.")
