"""
Calm3を用いて、2ターン目の応答を作成するスクリプト
"""

import copy
import json

from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# 入力ファイルと出力ファイルの設定
INPUT_FILE_NAME = "./hoge.jsonl"
BACKUP_FILE_NAME = "./backup.jsonl"
OUTPUT_FILE_NAME = "./hoge_answered.jsonl"

# モデルの設定
MODEL_NAME = "cyberagent/calm3-22b-chat"
TENSOR_PARALLEL_SIZE = 4  # 利用環境のGPUの数に合わせる
MAX_NUM_SEQS = 1000  # バッチサイズに合わせる？
MAX_NUM_BATCHED_TOKENS = 16384
DOWNLOAD_DIR = "./cache"

# サンプリングパラメータの設定
SAMPLING_PARAMS = SamplingParams(
    temperature=0.3,
    top_p=1,
    max_tokens=2048,
    repetition_penalty=1.1,
    stop=["<|im_start|>", "<|im_end|>", "<|im_sep|>", "</s>"],
)

# バッチ処理の設定
BATCH_SIZE = 1000  # バッチサイズ


def load_dataset(file_name):
    """データセットをファイルから読み込む"""
    with open(file_name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def initialize_model():
    """vLLMでモデルを初期化する"""
    return LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        download_dir=DOWNLOAD_DIR,
    )


def format_prompt(instruction, output, instruction2):
    """Calm3のprompt templateに整形する"""
    prompt = (
        "<|im_start|>system\nあなたは親切なAIアシスタントです。<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>\n"
        f"<|im_start|>user\n{instruction2}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def process_data(data_batch, model):
    """vLLMによるバッチ推論を使ったデータ合成"""
    prompts = [
        format_prompt(data["instruction"], data["output"], data["instruction2"])
        for data in data_batch
    ]
    outputs = model.generate(prompts, SAMPLING_PARAMS)

    results = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        if output.outputs[0].finish_reason == "stop":
            new_data = copy.deepcopy(data_batch[i])
            new_data["messages"].append({"role": "assistant", "content": text})
            new_data["output2"] = text
            results.append(new_data)

    return results


def save_backup(dataset, file_name="./backup.jsonl"):
    """バックアップを保存する関数"""
    with open(file_name, "w", encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def process_dataset(dataset, model, batch_size):
    """バッチサイズごとにデータを処理し、バックアップを保存する"""
    new_dataset = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        new_data_list = process_data(batch, model)
        new_dataset.extend(new_data_list)
        save_backup(new_dataset, BACKUP_FILE_NAME)
    return new_dataset


def main():
    """メイン処理"""
    dataset = load_dataset(INPUT_FILE_NAME)
    model = initialize_model()
    new_dataset = process_dataset(dataset, model, BATCH_SIZE)

    print(f"処理されたデータ数: {len(new_dataset)}")

    # 結果の保存
    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
        for item in new_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
