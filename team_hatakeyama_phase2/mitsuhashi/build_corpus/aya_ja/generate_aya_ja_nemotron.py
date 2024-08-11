# pip install tqdm
# pip install openai
# pip install datasets

import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset


# huggingface上の任意のデータセットを読み込む
# 文科省のテキストをソースにするのであればここを適切なファイルを読み込む処理に置き換える
#############################################
### 修正① データセットの読み込み
#############################################
dataset_name = 'ryota39/Aya_ja'
split = 'train'

# 出力先の指定
#############################################
### 修正② 出力先の指定
#############################################
out_dir = './out'
filename = f'{out_dir}/nemotron_'
os.makedirs(out_dir, exist_ok=True)

# deepinfraトークンを使ったOpenAIクライアントの作成
openai = OpenAI(
    api_key=os.environ.get('DEEP_INFRA'), # 共有されたdeepinfraのAPI_key
    base_url="https://api.deepinfra.com/v1/openai",
    )

# プロンプトの作成
# ここを用途に応じて修正する必要があります
#############################################
### 修正③ プロンプトの作成
#############################################
def build_prompt(record):
    return f'''次の質問に深呼吸してステップバイステップで答えなさい。答えは{record['targets']}であることが与えられています。初めに「答えは{record['targets']}です。」と回答した後、この答えに至る思考過程を述べて、最後に改めて「したがって、答えは{record['targets']}です。」と書きなさい。質問：{record['inputs']}\n'''

# deepinfraにリクエストを送信し、プロンプトを除いた生成文のみ返す
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


# データセットの読み込み
dataset = load_dataset(dataset_name)[split]

# 生成パラメータの指定
params = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    }


corpus = list()
tbar = tqdm(enumerate(dataset), total=len(dataset)) # 進捗バーの表示
for step, record in tbar:

    data_point = dict()

    # 先頭のn件だけ推論する場合
    # if step > 1:
    #     break

    if step < 6200:
        continue

    prompt = build_prompt(record)
    response = post_deep_infra(prompt, params)

    # データセットに必要なkeyとvalueを格納
    # ここのkey,valueは作るデータセットに応じて適宜設定
    #############################################
    ### 修正④ key, valueの作成
    #############################################
    data_point['idx'] = step
    data_point['targets'] = record['targets']
    data_point['inputs'] = record['inputs']
    data_point['language'] = record['language']
    data_point['annotation_type'] = record['annotation_type']
    data_point['language_code'] = record['language_code']
    data_point['user_id'] = record['user_id']
    data_point['response'] = response
    data_point['translation_prompt_choices'] ='''次の質問に深呼吸してステップバイステップで答えなさい。答えは{record['targets']}であることが与えられています。初めに「答えは{record['targets']}です。」と回答した後、この答えに至る思考過程を述べて、最後に改めて「したがって、答えは{record['targets']}です。」と書きなさい。質問：{record['inputs']}\n'''
    data_point['translation_model_choices'] = 'nvidia/Nemotron-4-340B-Instruct'

    # データセット1件分として格納
    corpus.append(data_point)

    # n件ごとに中間出力を保存する場合(n=100の場合)
    if not step == 0 and step % 100 == 0:
        with open(f'./out/aya_ja_{split}_{step}.json', mode='w', encoding="utf-8") as f:
            json.dump(corpus, f, indent=4, ensure_ascii=False)

# 全件分のデータ書き出し
with open(f'{filename}{split}.json', mode='w', encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)
