import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset


def build_prompt(record):
    return f'''次の質問に深呼吸してステップバイステップで回答しなさい。回答に「深呼吸」や「ステップバイステップ」という単語を含めてはいけません。回答は必ず日本語で回答しなさい。{record['inputs']}\n'''

def post_deep_infra(prompt, params):

    begin = time.perf_counter()
    completion = openai.chat.completions.create(
        model="nvidia/Nemotron-4-340B-Instruct",
        max_tokens=params['max_new_tokens'],
        temperature=params['temperature'],
        top_p=params['top_p'],
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content
    duration = time.perf_counter() - begin

    # print(f"duration: {duration:.04f}")
    return content


openai = OpenAI(
    api_key=os.environ.get('DEEP_INFRA'), # 共有されたdeepinfraのAPI_key
    base_url="https://api.deepinfra.com/v1/openai",
    )


filename = './out/aya_ja_nemotron_regen.json'
filename_out = './out/aya_ja_nemotron_regen_regen.json'

params = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    }

with open(filename, mode='r', encoding='utf-8') as f:
    records = json.load(f)

corpus = list()
tbar = tqdm(enumerate(records), total=len(records))
num_resolve = 0
for step, record in tbar:

    if step < 6200:
        continue

    regenerate = True if len(record['targets']) > 30 else False
    # regenerate = True if record['targets'] == 'この文章に含まれるキーワードとその解釈は以下です。' else False
    if regenerate:
        
        num_resolve += 1

        data_point = dict()

        # 先頭のn件だけ推論する場合
        # if step > 150:
        #     break

        prompt = build_prompt(record)
        response = post_deep_infra(prompt, params)

        data_point['idx'] = step
        data_point['targets'] = record['targets']
        data_point['inputs'] = record['inputs']
        data_point['language'] = record['language']
        data_point['annotation_type'] = record['annotation_type']
        data_point['language_code'] = record['language_code']
        data_point['user_id'] = record['user_id']
        data_point['response'] = response
        data_point['translation_prompt_choices'] = record['inputs']
        data_point['translation_model_choices'] = 'nvidia/Nemotron-4-340B-Instruct'

        # データセット1件分として格納
        corpus.append(data_point)
    else:
        corpus.append(record)

    # n件ごとに中間出力を保存する場合(n=100の場合)
    if not step == 0 and step % 100 == 0:
        with open(f'{filename_out}_{step}.json', mode='w', encoding="utf-8") as f:
            json.dump(corpus, f, indent=4, ensure_ascii=False)

with open(f'{filename_out}', mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)

print(f'num_resolve: {num_resolve}')
