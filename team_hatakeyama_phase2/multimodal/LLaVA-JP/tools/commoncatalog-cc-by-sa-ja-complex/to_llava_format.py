"""
commoncatalog-cc-by-sa-ja-complexの準備
python tools/commoncatalog-cc-by-sa-ja-complex/to_llava_format.py

========== count qa pairs ==========
question_ja 55005
====================================
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

INSTRUCTIONS = {
    "question_ja": None,
}

DATA_PATH = "./dataset/commoncatalog-cc-by-sa-ja-complex"


def create_llava_format(question, answer, image_filename):
    """
    Args:
        question (str): 質問プロンプト
        answer (str): 回答
        image_filename (str): 拡張子付きの画像ファイル名(image.jpgなど)
    """
    llava_format = {}
    conversations = []

    conversation_user = {"from": "ユーザー", "value": f"{question}\n<image>"}

    conversation_system = {"from": "システム", "value": answer}
    conversations.append(conversation_user)
    conversations.append(conversation_system)

    llava_format["image"] = image_filename
    llava_format["conversations"] = conversations

    # print(llava_format)

    return llava_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLAVA format dataset")
    args = parser.parse_args()

    dataset = load_dataset(
        "team-hatakeyama-phase2/commoncatalog-cc-by-sa-ja-complex", cache_dir="./cache"
    )

    llava_formats = {key: [] for key in INSTRUCTIONS.keys()}

    for data in dataset["train"]:
        image_filename = str(data["photoid"]) + "." + data["ext"]
        for key in INSTRUCTIONS.keys():
            answer = data["answer_ja"]

            # 文章が長過ぎるデータは除去
            if len(answer) <= 2000:
                llava_formats[key].append(
                    create_llava_format(data[key], answer, image_filename)
                )

    print("========== count qa pairs ==========")
    for key, item in llava_formats.items():
        print(key, len(item))
    print("====================================")

    for key in INSTRUCTIONS.keys():
        output_filename = key

        caption_path = Path(
            DATA_PATH, output_filename + ".json"
        )

        with open(caption_path, mode="w") as f:
            json.dump(llava_formats[key], f, indent=2, ensure_ascii=False)
