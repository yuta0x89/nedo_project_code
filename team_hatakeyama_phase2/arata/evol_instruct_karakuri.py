"""
karakuri-instructとEvol-Instructを用いて、単一の対話データから複数バリエーションの対話データを作成するスクリプト
"""

import copy
import json
import random

from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# ファイルパスの設定
INPUT_FILE = "./hoge.jsonl"
OUTPUT_BACKUP_FILE = "./backup.jsonl"
OUTPUT_FILE = "./hoge_evolved.jsonl"

# モデル設定
MODEL_NAME = "karakuri-ai/karakuri-lm-8x7b-instruct-v0.1"
TENSOR_PARALLEL_SIZE = 4  # 利用環境のGPUの数に合わせる
MAX_NUM_SEQS = 1000  # バッチサイズに合わせる？
MAX_NUM_BATCHED_TOKENS = 32768
DOWNLOAD_DIR = "./cache"

# サンプリングパラメータの設定
SAMPLING_PARAMS = SamplingParams(
    temperature=0.5,
    top_p=0.9,
    max_tokens=1024,
    repetition_penalty=1.1,
    stop=["[INST]", "[/INST]", "</s>"],
)

# バッチ処理の設定
BATCH_SIZE = 1000  # バッチサイズを指定

# ベースとなる指示文
BASE_INSTRUCTION = "あなたの役割は質問文・プロンプトの修正です。\n\
あなたの目的は、与えられた質問文・プロンプトをより複雑で難しいバージョンに書き換えることです。\n\
ただし、書き換えられた質問は合理的で、人間が理解して応答できるものでなければならず、より自然な日本語でなければなりません。\n\
あなたの書き換えは、#与えられた質問#:の表やコードなどのテキスト以外の部分を省略してはいけません。また、#与えられたプロンプト#の入力を省略しないでください。\n\
あなたは以下の方法を使って与えられた質問を複雑にする必要があります：\n\
{} \n\
#書き換えられた質問#が冗長になりすぎないように最善を尽くしてください。#書き換えられた質問#は#与えられた質問#に30〜40語のみ追加できます。\n\
#書き換えられた質問#で1つの文を長くしすぎないようにしてください。長い文は読みやすくするために複数の文に分割する必要があります。\n\
'#与えられた質問#'、'#書き換えられた質問#'、'与えられた質問'、'書き換えられた質問'は#書き換えられた質問#に表示されることは許可されていません。質問には答えず、書き換えた新たな質問やプロンプトのみを出力してください。\n"
BASE_DEPTH_INPUT_INSTRUCTION = "あなたの役割は質問文・プロンプトの修正です。\n\
あなたの目的は、データ形式を使用して与えられた質問文・プロンプトをより複雑で難しいバージョンに書き換えることです。\n\
ただし、書き換えられた質問は合理的で、人間が理解して応答できるものでなければならず、より自然な日本語でなければなりません。\n\
#書き換えられた質問#に{}形式のテキストを入力データとして追加する必要があります。\n\
#書き換えられた質問#が冗長になりすぎないように最善を尽くしてください。#書き換えられた質問#は#与えられたプロンプト#に30〜40語のみ追加できます。\n\
#書き換えられた質問#で1つの文を長くしすぎないようにしてください。長い文は読みやすくするために複数の文に分割する必要があります。\n\
'#与えられた質問#'、'#書き換えられた質問#'、'与えられた質問'、'書き換えられた質問'は#書き換えられた質問#に表示されることは許可されていません。質問には答えず、書き換えた新たな質問やプロンプトのみを出力してください。\n"
BASE_INSTRUCTION_BREATH = "あなたの役割は質問文・プロンプトの作成です。\n\
あなたの目標は、#与えられた質問#からインスピレーションを得て、Pythonに関連するプロンプトを他のプログラミングに関連するプロンプトに変更するなど、完全に新しい質問文・プロンプトを作成することです。\n\
この新しいプロンプトはより珍しいものでなければならず、より自然な日本語でなければなりません。\n\
#書き換えられた質問#の長さと複雑さは、#与えられた質問#と同程度か、少し難しくする必要があります。\n\
#書き換えられた質問#は合理的で、人間が容易に理解して応答できるものでなければならず、より自然な日本語でなければなりません。\n\
'#与えられた質問#'、'#書き換えられた質問#'、'与えられた質問'、'書き換えられた質問'は#書き換えられた質問#に表示されることは許可されていません。質問には答えず、書き換えた新たな質問やプロンプトのみを出力してください。\n"


def load_dataset(file_path):
    """データセットをファイルから読み込む"""
    with open(file_path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f.readlines()
            if json.loads(line)["language"] == "Japanese"
        ]


def initialize_model():
    """vLLMモデルの初期化"""
    return LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        download_dir=DOWNLOAD_DIR,
    )


# 書き換え用のプロンプト作成
def create_constraints_prompt(instruction):
    system_prompt = BASE_INSTRUCTION.format(
        "#与えられた質問#にもう1つの制約/要件を追加してください。"
    )
    user_prompt = f"#与えられた質問#:\n {instruction}"
    return system_prompt, user_prompt


def create_deepen_prompt(instruction):
    system_prompt = BASE_INSTRUCTION.format(
        "もし#与えられた質問#に特定の問題に関する質問が含まれている場合、その質問の深さと広さを増やしてください。"
    )
    user_prompt = f"#与えられた質問#:\n {instruction}"
    return system_prompt, user_prompt


def create_concretizing_prompt(instruction):
    system_prompt = BASE_INSTRUCTION.format(
        "一般的な概念をより具体的な概念に置き換えてください。"
    )
    user_prompt = f"#与えられた質問#:\n {instruction}"
    return system_prompt, user_prompt


def create_reasoning_prompt(instruction):
    system_prompt = BASE_INSTRUCTION.format(
        "もし#与えられた質問#が単純な思考プロセスだけで解決できる場合、複数のステップの推論を明示的に要求するように書き換えてください。"
    )
    user_prompt = f"#与えられた質問#:\n {instruction}"
    return system_prompt, user_prompt


def create_complicate_input_prompt(instruction, data_format):
    system_prompt = BASE_DEPTH_INPUT_INSTRUCTION.format(data_format)
    user_prompt = f"#与えられた質問#:\n {instruction}"
    return system_prompt, user_prompt


def create_breadth_prompt(instruction):
    system_prompt = BASE_INSTRUCTION_BREATH
    user_prompt = f"#与えられた質問#:\n {instruction}"
    return system_prompt, user_prompt


def select_input_data():
    data_formats = [
        "XML data",
        "SQL database",
        "python code",
        "HTML page",
        "Shell cmd",
        "JSON data",
    ]
    return random.choice(data_formats)


def select_evolution_prompt(instruction):
    evolution_prompts = [
        {
            "system_prompt": create_constraints_prompt(instruction)[0],
            "user_prompt": create_constraints_prompt(instruction)[1],
            "evol_type": "constraints",
        },
        {
            "system_prompt": create_deepen_prompt(instruction)[0],
            "user_prompt": create_deepen_prompt(instruction)[1],
            "evol_type": "deepen",
        },
        {
            "system_prompt": create_concretizing_prompt(instruction)[0],
            "user_prompt": create_concretizing_prompt(instruction)[1],
            "evol_type": "concretizing",
        },
        {
            "system_prompt": create_reasoning_prompt(instruction)[0],
            "user_prompt": create_reasoning_prompt(instruction)[1],
            "evol_type": "reasoning",
        },
        {
            "system_prompt": create_complicate_input_prompt(
                instruction, select_input_data()
            )[0],
            "user_prompt": create_complicate_input_prompt(
                instruction, select_input_data()
            )[1],
            "evol_type": "complicate_input",
        },
    ]
    evolution_prompts.extend(
        [
            {
                "system_prompt": create_breadth_prompt(instruction)[0],
                "user_prompt": create_breadth_prompt(instruction)[1],
                "evol_type": "breadth",
            }
            for _ in range(3)
        ]
    )
    return random.choice(evolution_prompts).values()


def format_prompt(system_prompt, user_prompt):
    """karakuri-instructのprompt templateに整形する"""
    prompt = (
        f"<s><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|>"
        f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><attributes>helpfulness: 4 correctness: 4 coherence: 4 complexity: 4 verbosity: 4 quality: 4 toxicity: 0 humor: 0 creativity: 0</attributes>#書き換えられた質問#:\n"
    )
    return prompt


def process_data(data_batch, model):
    results = [[] for _ in range(5)]  # 5回の進化結果を格納するリスト

    for step in range(5):
        loop_data = data_batch if step == 0 else results[step - 1]
        prompts = []

        for i, data in enumerate(loop_data):
            if step == 0:
                instruction = data["messages"][0]["content"]
            else:
                try:
                    instruction = results[step - 1][i]["messages"][0]["content"]
                except IndexError:
                    continue  # エラーが発生した場合、そのデータをスキップ

            system_prompt, user_prompt, evol_type = select_evolution_prompt(instruction)
            data["evol_history"].append(evol_type)
            prompts.append(format_prompt(system_prompt, user_prompt))

        outputs = model.generate(prompts, SAMPLING_PARAMS)

        for i, (data, output) in enumerate(zip(loop_data, outputs)):
            if output.outputs[0].finish_reason == "stop":
                new_data = copy.deepcopy(data)
                new_data["messages"][0]["content"] = output.outputs[0].text.strip()
                results[step].append(new_data)

    return results


def process_dataset(dataset, model, batch_size=10):
    for data in dataset:
        data["evol_history"] = []

    all_results = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        batch_results = process_data(batch, model)
        all_results.extend(zip(*batch_results))  # 各データの5回の進化結果をまとめる
        save_backup(all_results)
    return all_results


def save_backup(dataset):
    new_dataset = [item for sublist in dataset for item in sublist]
    with open(OUTPUT_BACKUP_FILE, "w", encoding="utf-8") as f:
        for item in new_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def lacks_punctuation(s):
    return not any(char in s for char in ["。", "？", "?", "."])


def contains_bad_word(s):
    bad_words = ["プロンプト"]
    return any(word in s for word in bad_words)


def filter_data(dataset):
    filtered_data = []
    for i, data in enumerate(dataset):
        content = data["messages"][0]["content"]
        if (
            len(content) <= 10
            or lacks_punctuation(content)
            or contains_bad_word(content)
        ):
            continue
        data["evol_model"] = MODEL_NAME
        data["evol_generation"] = len(data["evol_history"])
        data["original_id"] = data["id"]
        data["messages"] = [data["messages"][0]]
        data["instruction"] = content
        if "code_language" in data:
            del data["code_language"]
        filtered_data.append(data)
    return filtered_data


def main():
    dataset = load_dataset(INPUT_FILE)
    model = initialize_model()
    all_results = process_dataset(dataset, model, batch_size=BATCH_SIZE)
    new_dataset = filter_data([item for sublist in all_results for item in sublist])

    for i, data in enumerate(new_dataset):
        data["id"] = i

    print(f"作成されたデータ数: {len(new_dataset)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in new_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
