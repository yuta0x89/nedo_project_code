"""
Qwen2とEvol-Instructを用いて、単一の対話データから複数バリエーションの対話データを作成するスクリプト
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
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
TENSOR_PARALLEL_SIZE = 4  # 利用環境のGPUの数に合わせる
MAX_NUM_SEQS = 1000  # バッチサイズに合わせる？
MAX_NUM_BATCHED_TOKENS = 32768
DOWNLOAD_DIR = "./cache"

# サンプリングパラメータの設定
SAMPLING_PARAMS = SamplingParams(
    temperature=1,
    top_p=1,
    max_tokens=1024,
    repetition_penalty=1.1,
    stop=["<|im_start|>", "<|im_end|>", "<|im_sep|>", "</s>"],
)

# バッチ処理の設定
BATCH_SIZE = 1000  # バッチサイズを指定

# ベースとなる指示文
BASE_INSTRUCTION = "I want you act as a Prompt Rewriter.\n\
Your objective is to rewrite a given prompt into a more complex and harder version.\n\
But the rewritten prompt must be reasonable, can be understood and responded to by humans, and must be more natural English.\n\
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \n\
You SHOULD complicate the given prompt using the following method: \n\
{} \n\
You should try your best not to make the #Rewritten Prompt# become too verbose, #Rewritten Prompt# can only add 30 to 40 words into #The Given Prompt#. \n\
You should try not to make one sentence too long in #Rewritten Prompt#. Long sentences should be broken up into multiple sentences to keep them readable. \n\
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n"
BASE_DEPTH_INPUT_INSTRUCTION = "I want you act as a Prompt Rewriter.\n\
Your objective is to rewrite a given prompt into a more complex and harder version using dataformat.\n\
But the rewritten prompt must be reasonable and must be understood and responded by humans, and must be more natural English.\n\
You must add {} format text as input data in #Rewritten Prompt#.\n\
You should try your best not to make the #Rewritten Prompt# become too verbose, #Rewritten Prompt# can only add 30 to 40 words into #The Given Prompt#. \n\
You should try not to make one sentence too long in #Rewritten Prompt#. Long sentences should be broken up into multiple sentences to keep them readable. \n\
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n"
BASE_INSTRUCTION_BREATH = "I want you act as a Prompt Creator.\n\
Your goal is to draw inspiration from the #Given Prompt# to create a completely brand new prompt like changing python-related prompt to or other programming related prompt.\n\
This new prompt should be much rarer and it must be more natural English.\n\
The LENGTH and complexity of the #Rewritten Prompt# should be similar or a bit harder to that of the #Given Prompt#.\n\
The #Rewritten Prompt# must be reasonable, easily understood and responded to by humans, must be more natural English.\n\
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n"


def load_dataset(file_path):
    """データセットをファイルから読み込む"""
    with open(file_path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f.readlines()
            if json.loads(line)["language"] == "English"
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
        "Please add one more constraints/requirements into #The Given Prompt#"
    )
    user_prompt = f"#The Given Prompt#:\n {instruction}"
    return system_prompt, user_prompt


def create_deepen_prompt(instruction):
    system_prompt = BASE_INSTRUCTION.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
    )
    user_prompt = f"#The Given Prompt#:\n {instruction}"
    return system_prompt, user_prompt


def create_concretizing_prompt(instruction):
    system_prompt = BASE_INSTRUCTION.format(
        "Please replace general concepts with more specific concepts."
    )
    user_prompt = f"#The Given Prompt#:\n {instruction}"
    return system_prompt, user_prompt


def create_reasoning_prompt(instruction):
    system_prompt = BASE_INSTRUCTION.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    )
    user_prompt = f"#The Given Prompt#:\n {instruction}"
    return system_prompt, user_prompt


def create_complicate_input_prompt(instruction, data_format):
    system_prompt = BASE_DEPTH_INPUT_INSTRUCTION.format(data_format)
    user_prompt = f"#The Given Prompt#:\n {instruction}"
    return system_prompt, user_prompt


def create_breadth_prompt(instruction):
    system_prompt = BASE_INSTRUCTION_BREATH
    user_prompt = f"#The Given Prompt#:\n {instruction}"
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
    """Qwen2のprompt templateに整形する"""
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant. You must answer only in natural English.<|im_end|>\n"
        f"<|im_start|>user\n{system_prompt}{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n#Rewritten Prompt#:\n\n"
    )
    return prompt


def process_data(data_batch, model):
    results = [[] for _ in range(3)]  # 3回の進化結果を格納するリスト

    for step in range(3):
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
    bad_words = ["#", "prompt"]
    return all(word in s for word in bad_words)


# Qwen2は英語以外が混ざりやすいのでフィルタする
def is_not_english(text):
    for char in text:
        code_point = ord(char)
        if (
            0x4E00 <= code_point <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code_point <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0x20000 <= code_point <= 0x2A6DF  # CJK Unified Ideographs Extension B
            or 0x2A700 <= code_point <= 0x2B73F  # CJK Unified Ideographs Extension C
            or 0x2B740 <= code_point <= 0x2B81F  # CJK Unified Ideographs Extension D
            or 0x2B820 <= code_point <= 0x2CEAF  # CJK Unified Ideographs Extension E
            or 0x2CEB0 <= code_point <= 0x2EBEF  # CJK Unified Ideographs Extension F
            or 0x30000 <= code_point <= 0x3134F  # CJK Unified Ideographs Extension G
            or 0x1100 <= code_point <= 0x11FF  # Hangul Jamo
            or 0x2E80 <= code_point <= 0x2EFF  # CJK Radicals Supplement
            or 0x2F00 <= code_point <= 0x2FDF  # Kangxi Radicals
            or 0x3000 <= code_point <= 0x303F  # CJK Symbols and Punctuation
            or 0x3040 <= code_point <= 0x309F  # Hiragana
            or 0x30A0 <= code_point <= 0x30FF  # Katakana
            or 0x3100 <= code_point <= 0x312F  # Bopomofo
            or 0x3130 <= code_point <= 0x318F  # Hangul Compatibility Jamo
            or 0x3190 <= code_point <= 0x319F  # Kanbun
            or 0x31A0 <= code_point <= 0x31BF  # Bopomofo Extended
            or 0x31C0 <= code_point <= 0x31EF  # CJK Strokes
            or 0x31F0 <= code_point <= 0x31FF  # Katakana Phonetic Extensions
            or 0x3200 <= code_point <= 0x32FF  # Enclosed CJK Letters and Months
            or 0x3300 <= code_point <= 0x33FF  # CJK Compatibility
            or 0xA960 <= code_point <= 0xA97F  # Hangul Jamo Extended-A
            or 0xAC00 <= code_point <= 0xD7A3  # Hangul Syllables
            or 0xD7B0 <= code_point <= 0xD7FF  # Hangul Jamo Extended-B
            or 0xF900 <= code_point <= 0xFAFF  # CJK Compatibility Ideographs
            or 0xFE30 <= code_point <= 0xFE4F  # CJK Compatibility Forms
            or 0xFF00 <= code_point <= 0xFFEF  # Halfwidth and Fullwidth Forms
            or 0x1B000 <= code_point <= 0x1B0FF  # Kana Supplement
            or 0x1B100 <= code_point <= 0x1B12F  # Kana Extended-A
            or 0x1B130 <= code_point <= 0x1B16F  # Small Kana Extension
            or 0x1F200 <= code_point <= 0x1F2FF  # Enclosed Ideographic Supplement
            or 0x2F800
            <= code_point
            <= 0x2FA1F  # CJK Compatibility Ideographs Supplement
        ):
            return True
    return False


def filter_data(dataset):
    filtered_data = []
    for i, data in enumerate(dataset):
        content = data["messages"][0]["content"]
        if (
            len(content) <= 10
            or lacks_punctuation(content)
            or contains_bad_word(content)
            or is_not_english(content)
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
